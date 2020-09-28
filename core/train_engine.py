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

import torchph.pershom as pershom

from torch.nn import Sequential, Linear, ReLU
from torch.optim.lr_scheduler import MultiStepLR

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool

from torchph import pershom
ph = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch

from torchph.nn import SLayerRationalHat
from collections import defaultdict, Counter

from .model import PershomLearnedFilt, PershomRigidDegreeFilt, GIN, SimpleNNBaseline
from .data import dataset_factory, train_test_val_split
from .utils import my_collate, evaluate


import torch.multiprocessing as mp
# try:
#      mp.set_start_method('spawn')
# except RuntimeError:
#     pass


__training_cfg = {
    'lr': float,
    'lr_drop_fact': float,
    'num_epochs': int,
    'epoch_step': int,
    'batch_size': int,
    'weight_decay': float,
    'validation_ratio': float,
}


__model_cfg_meta = {
    'model_type': str,
    'use_super_level_set_filtration': bool,
    'use_node_degree': bool,
    'set_node_degree_uninformative': bool,
    'pooling_strategy': str,
    'use_node_label': bool,
    'gin_number': int,
    'gin_dimension': int,
    'gin_mlp_type': str,
    'num_struct_elements': int,
    'cls_hidden_dimension': int,
    'drop_out': float,
}


__exp_cfg_meta = {
    'dataset_name': str,
    'training': __training_cfg,
    'model': __model_cfg_meta,
    'tag': str
}


__exp_res_meta = {
    'exp_cfg': __exp_cfg_meta,
    'cv_test_acc': list,
    'cv_val_acc': list,
    'cv_indices_trn_tst_val': list,
    'cv_epoch_loss': list,
    'start_time': list,
    'id': str
}


def model_factory(model_cfg: dict, dataset):
    str_2_type = {
        'PershomRigidDegreeFilt': PershomRigidDegreeFilt,
        'PershomLearnedFilt': PershomLearnedFilt,
        'GIN': GIN, 
        'SimpleNNBaseline': SimpleNNBaseline
    }

    model_type = model_cfg['model_type']
    Model = str_2_type[model_type]
    return Model(dataset, **model_cfg)


def experiment(exp_cfg, device, output_dir=None, verbose=True, output_cache=None):

    training_cfg = exp_cfg['training']

    model_cfg = exp_cfg['model']

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = dataset_factory(exp_cfg['dataset_name'], verbose=verbose)
    split_ds, split_i = train_test_val_split(
        dataset,
        validation_ratio=training_cfg['validation_ratio'],
        verbose=verbose)

    cv_test_acc = [[] for _ in range(len(split_ds))]
    cv_val_acc = [[] for _ in range(len(split_ds))]
    cv_epoch_loss = [[] for _ in range(len(split_ds))]

    uiid = str(uuid.uuid4())

    if output_dir is not None:
        output_path = osp.join(output_dir, uiid + '.pickle')

    ret = {} if output_cache is None else output_cache

    ret['exp_cfg'] = exp_cfg
    ret['cv_test_acc'] = cv_test_acc
    ret['cv_val_acc']  = cv_val_acc
    ret['cv_indices_trn_tst_val'] = split_i
    ret['cv_epoch_loss'] = cv_epoch_loss
    ret['start_time'] = str(datetime.datetime.now())
    ret['id'] = uiid
    ret['finished_training'] = False

    for fold_i, (train_split, test_split, validation_split) in enumerate(split_ds):

        model = model_factory(model_cfg, dataset).to(device)

        if verbose and fold_i == 0:
            print(model)

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
            shuffle=True,
            # if last batch would have size 1 we have to drop it ...
            drop_last=(len(train_split) % training_cfg['batch_size'] == 1)
        )

        dl_test = torch.utils.data.DataLoader(
            test_split ,
            collate_fn=my_collate,
            batch_size=64,
            shuffle=False
        )

        dl_val = None
        if training_cfg['validation_ratio'] > 0:
            dl_val = torch.utils.data.DataLoader(
                validation_split,
                collate_fn=my_collate,
                batch_size=64,
                shuffle=False
            )

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

                if verbose:
                    print("Epoch {}/{}, Batch {}/{}".format(
                        epoch_i,
                        training_cfg['num_epochs'],
                        batch_i,
                        len(dl_train)),
                        end='\r')

                # break # todo remove!!!

            if verbose: print('')

            test_acc = evaluate(dl_test, model, device)
            cv_test_acc[fold_i].append(test_acc*100.0)
            cv_epoch_loss[fold_i].append(epoch_loss)

            val_acc = None
            if training_cfg['validation_ratio'] > 0.0:
                val_acc = evaluate(dl_val, model, device)
                cv_val_acc[fold_i].append(val_acc*100.0)

            if verbose: print("loss {:.2f} | test_acc {:.2f} | val_acc {:.2f}".format(epoch_loss, test_acc*100.0, val_acc*100.0))

        # break #todo remove!!!

        if output_dir is not None:
            model_file = osp.join(output_dir, uiid + '_model_{}.pht'.format(fold_i))
            torch.save(model.to('cpu'), model_file)

            with open(output_path, 'bw') as fid:
                pickle.dump(file=fid, obj=ret)

    ret['finished_training'] = True
    if output_dir is not None:
        with open(output_path, 'bw') as fid:
            pickle.dump(file=fid, obj=ret)

    return ret


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
        print(exp_cfg['dataset_name'])
        experiment(exp_cfg, device, output_dir=output_dir, verbose=False)
        device_counter[device_id] -= 1

    except Exception as ex:
        ex.exp_cfg = exp_cfg
        device_counter[device_id] -= 1

        return ex


def experiment_multi_device(exp_cfgs, output_dir, visible_devices, max_process_on_device):
    assert isinstance(exp_cfgs, list)
    assert isinstance(visible_devices, list)
    assert osp.isdir(output_dir)
    assert all((i < torch.cuda.device_count() for i in visible_devices))

    num_device = len(visible_devices)

    manager = mp.Manager()
    device_counter = manager.dict({t: 0 for t in visible_devices})
    lock = manager.Lock()

    task_args = [(exp_cfg, output_dir, device_counter, lock, max_process_on_device) for exp_cfg in exp_cfgs]

    ret = []
    with mp.Pool(num_device*max_process_on_device, maxtasksperchild=1) as pool:

        for i, r in enumerate(pool.imap_unordered(experiment_task, task_args)):
            ret.append(r)

            if r is None:
                print("# Finished job {}/{}".format(i + 1, len(task_args)))

            else:
                print("#")
                print("# Error in job {}/{}".format(i, len(task_args)))
                print("#")
                print("# Error:")
                print(r)
                print("# experiment configuration:")
                print(r.exp_cfg)

    ret = [r for r in ret if r is not None]
    if len(ret) > 0:
        with open(osp.join(output_dir, 'errors.pickle'), 'bw') as fid:
            pickle.dump(obj=ret, file=fid)
