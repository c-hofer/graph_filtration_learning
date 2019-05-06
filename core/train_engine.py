import os.path as osp
import itertools
import copy
import uuid
import pickle
import datetime

import torch
import torch.nn.functional as F
import torch_geometric

import numpy as np

import torch.nn as nn
import torch.optim as optim

import chofer_torchex.pershom as pershom

from torch.nn import Sequential, Linear, ReLU
from torch.optim.lr_scheduler import MultiStepLR

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool

from chofer_torchex import pershom
ph = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch

from chofer_torchex.nn import SLayerRationalHat
from collections import defaultdict, Counter

import core.model
from .data import dataset_factory, train_test_split
from .utils import my_collate, evaluate


import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass


def experiment_task(args):

    exp_cfg, output_dir, device_counter, lock, max_process_on_device = args
    
    with lock:
        device = None
        for k, v in device_counter.items():
            if v < max_process_on_device:
                device_id = k
                device = 'cuda:{}'.format(device_id)

                break            
        device_counter[device_id] += 1
    
    assert device is not None

    try:
        experiment(exp_cfg, output_dir, device, verbose=False)
        device_counter[device_id] -= 1

    except Exception as ex:
        ex.exp_cfg = exp_cfg
        device_counter[device_id] -= 1
        raise ex


def experiment(exp_cfg, output_dir, device, verbose=True):

    model_id_to_class = {
        'PershomModel': core.model.PershomModel
    }      
    
    training_cfg = exp_cfg['training']
    model_cfg = exp_cfg['model']

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = dataset_factory(exp_cfg['dataset_name'], verbose=verbose)
    splits = train_test_split(dataset, verbose=verbose)

    cv_acc = [[] for _ in range(len(splits))]
    cv_epoch_loss = [[] for _ in range(len(splits))]
    
    output_path = osp.join(output_dir,str(uuid.uuid4()) + '.pickle')
    ret = {
        'exp_cfg': exp_cfg, 
        'cv_acc': cv_acc, 
        'cv_epoch_loss': cv_epoch_loss, 
        'start_time': datetime.datetime.now()
    }
    
    for fold_i, (train_split, test_split) in enumerate(splits):      
        

        model = model_id_to_class[model_cfg['model_type']](
            dataset, 
            **model_cfg['model_kwargs']
        ).to(device)

        opt = optim.Adam(
            model.parameters(), 
            lr=training_cfg['lr'], 
            weight_decay=training_cfg['weight_decay']
        )

        scheduler = MultiStepLR(opt, 
                                milestones=list(range(0, 
                                                      training_cfg['num_epochs'], 
                                                      training_cfg['epoch_step'])
                                               )[1:],
                                gamma=training_cfg['lr_drop_fact'])

        dl_train = torch.utils.data.DataLoader(
            train_split, 
            collate_fn=my_collate, 
            batch_size=training_cfg['batch_size'], 
            shuffle=True)

        dl_test = torch.utils.data.DataLoader(
            test_split , 
            collate_fn=my_collate, 
            batch_size=64, 
            shuffle=False)

        for epoch_i in range(1, training_cfg['num_epochs']+1):

            model.train()
            scheduler.step()
            epoch_loss = 0

            for batch_i, batch in enumerate(dl_train, start=1):  

                batch = batch.to(device)
                if not hasattr(batch, 'node_lab'): batch.node_lab = None
                batch.boundary_info = [e.to(device) for e in batch.boundary_info]

                y_hat = model(batch)

                loss = torch.nn.functional.cross_entropy(y_hat, batch.y)
                opt.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                opt.step()

            test_acc = evaluate(dl_test, model, device)
            if verbose: print("loss {:.2f} | test_acc {:.2f}".format(epoch_loss, test_acc*100.0))
            
            cv_acc[fold_i].append(test_acc*100.0)
            cv_epoch_loss[fold_i].append(epoch_loss)
    
        with open(output_path, 'bw') as fid:
            pickle.dump(file=fid, obj=ret)   


def experiment_multi_device(exp_cfgs, output_dir, visible_devices, max_process_on_device):
    assert isinstance(exp_cfgs, list)
    assert isinstance(visible_devices, list) 
    assert osp.isdir(output_dir)
    assert all((i < torch.cuda.device_count() for i in visible_devices))

    num_device = len(visible_devices)
    max_process_on_device = max_process_on_device

    manager = mp.Manager()
    device_counter = manager.dict({t: 0 for t in visible_devices})
    lock = manager.Lock()

    task_args = [(exp_cfg, output_dir, device_counter, lock, max_process_on_device) for exp_cfg in exp_cfgs]

    ret = []
    with mp.Pool(num_device*max_process_on_device, maxtasksperchild=1) as pool:

        for i, r in enumerate(pool.imap_unordered(experiment_task, task_args)):
            ret.append(r)

            if r is None: 
                print("# Finished job {}/{}".format(i, len(task_args)))

            else:
                print("#")
                print("# Error in job {}/{}".format(i, len(task_args)))
                print("#")
                print("# Error:")
                print(r)
                print("# experiment configuration:")
                print(r.exp_cfg)


if __name__ == '__main__':
    dataset_names = [
        #'COLLAB',
        #'IMDB-MULTI',
        'IMDB-BINARY',
        #'ENZYMES']
        #'PTC_PGNN',
        #'PTC_FM',
        #'PTC_FR',
        #'PTC_MM',
        #'PTC_MR',
        #'PROTEINS',
        #'DD',
        # 'NCI1']
        #'MUTAG'
    ]
    print(dataset_names)


    dataset_has_node_lab = {n: True if dataset_factory(n, verbose=False).num_node_lab is not None else False for n in dataset_names}
    dataset_has_node_lab
    
    training_cfg = {
        'lr': 0.01, 
        'lr_drop_fact': 0.5, 
        'num_epochs': 60,
        'epoch_step': 10,
        'batch_size': 64,
        'weight_decay': 10e-06,
    }
    training_cfgs = [training_cfg]

    model_cfg = {
        'model_type': 'PershomModel',
        'model_kwargs': {
            'filtration_kwargs': {
                'use_node_deg': None,
                'use_node_lab': None,
                'num_gin': 1,
                'hidden_dim': 64, 
                'use_mlp': None
            }
        }
    }

    model_cfgs = []
    for a, b, c in itertools.product([True, False], [True, False], [True, False]):
        if (not a) and (not b):
            continue
        
        tmp = copy.deepcopy(model_cfg)
        tmp['model_kwargs']['filtration_kwargs']['use_node_deg'] = a
        tmp['model_kwargs']['filtration_kwargs']['use_node_lab'] = b
        tmp['model_kwargs']['filtration_kwargs']['use_mlp'] = c
        
        model_cfgs.append(tmp)
        
    no_filt_learning = copy.deepcopy(model_cfg)
    no_filt_learning['model_kwargs']['filtration_kwargs'] = None

    model_cfgs.append(no_filt_learning)

    exp_cfgs = []
    for a, b, c in itertools.product(dataset_names, training_cfgs, model_cfgs):
        
        # filter out datasets which have no node labels
        if not dataset_has_node_lab[a]:
            if c['model_kwargs']['filtration_kwargs'] is not None:
                use_node_lab = c['model_kwargs']['filtration_kwargs']['use_node_lab']
                
                if use_node_lab:
                    continue
                
        
        tmp = {
            'dataset_name': a, 
            'training': b, 
            'model': c
        }
        exp_cfgs.append(tmp)
        
    print("# Number of configurations", len(exp_cfgs))

    #output_dir = '/home/pma/chofer/repositories/nips_2019_code/results'
    output_dir = '/tmp/test_output_dir'


    num_device = 4
    max_process_on_device = 2
    manager = mp.Manager()
    device_counter = manager.dict({t: 0 for t in range(num_device)})
    lock = manager.Lock()

    task_args = [(exp_cfg, output_dir, device_counter, lock, max_process_on_device) for exp_cfg in exp_cfgs]

    with mp.Pool(num_device*max_process_on_device, maxtasksperchild=1) as pool:
        for i, _ in enumerate(pool.imap_unordered(experiment, task_args)):
            print("# Finished job {}/{}".format(i, len(task_args)))
