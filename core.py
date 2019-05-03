import os.path as osp
import itertools
import torch
import torch_geometric
import torch_geometric.data
import numpy as np

import chofer_torchex.pershom as pershom

from torch_geometric.datasets import TUDataset
from collections import Counter
from sklearn.model_selection import StratifiedKFold

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def ph(batch):
    ret = []
    with torch.cuda.device(batch[0][0].device):
        for b in batch:
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                ret.append(pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_single(*b))
            
        torch.cuda.synchronize()
            
    return ret

ph = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        assert isinstance(X, list)
        self.data = X
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for i in range(len(self.data)):
            yield self.data[i]
    
    
class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.ds = dataset
        self.indices = indices
        
        assert len(indices) <= len(dataset)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]
    
    
def my_collate(data_list):
    ret = torch_geometric.data.Batch().from_data_list(data_list)
    
    boundary_info = []
    sample_pos = [0]
    for d in data_list:
        boundary_info.append(d.boundary_info)
        sample_pos.append(d.num_nodes)
     
    ret.sample_pos = torch.tensor(sample_pos).cumsum(0)
    ret.boundary_info = boundary_info
    
    return ret


POWERFUL_GNN_DATASET_NAMES =  ["PTC_PGNN"]
TU_DORTMUND_DATASET_NAMES = [
    "NCI1", "PTC_MR", "PROTEINS", 
    "REDDIT-BINARY", "REDDIT-MULTI-5K", 
    "ENZYMES", "DD", "IMDB-BINARY", "IMDB-MULTI", "MUTAG"
]


def load_powerfull_gnn_dataset(dataset_name):
    has_node_features = []
    assert dataset_name in POWERFUL_GNN_DATASET_NAMES
    
    dataset_name = "PTC"
    path = "/home/pma/chofer/repositories/powerful-gnns/dataset/{}/{}.txt".format(dataset_name, dataset_name)
    has_node_features = dataset_name in has_node_features

    with open(path, 'r') as f:
        num_graphs = int(f.readline().strip())

        data = []

        graph_label_map = {}
        node_label_map = {}

        for i in range(num_graphs):
            row = f.readline().strip().split()
            num_nodes, graph_label = [int(w) for w in row]

            if graph_label not in graph_label_map:
                graph_label_map[graph_label] = len(graph_label_map)

            graph_label = graph_label_map[graph_label]


            nodes = []
            node_labels = []
            edges = []
            node_features = []

            for node_id in range(num_nodes):
                nodes.append(node_id)

                row = f.readline().strip().split()

                node_label = int(row[0])

                if node_label not in node_label_map:
                    node_label_map[node_label] = len(node_label_map)

                node_labels.append(node_label_map[node_label])

                num_neighbors = int(row[1])            
                neighbors = [int(i) for i in row[2:num_neighbors+2]]
                assert num_neighbors == len(neighbors)

                edges += [(node_id, neighbor_id) for neighbor_id in neighbors]

                if has_node_features:                
                    node_features = [float(i) for i in row[(2 + num_neighbors):]]
                    assert len(node_features) != 0           

            x = torch.tensor(node_features) if has_node_features else None

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_index = edge_index.permute(1, 0)
            tmp = edge_index.index_select(0, torch.tensor([1, 0]))
            edge_index = torch.cat([edge_index, tmp], dim=1)

            y = torch.tensor([graph_label])

            data.append(
                torch_geometric.data.Data(
                    x=x, 
                    edge_index=edge_index, 
                    y = y
                )
            )  

    ds = SimpleDataset(data)
    ds.name = dataset_name
    return ds
    

def get_boundary_info(g):
    
    e = g.edge_index.permute(1, 0).sort(1)[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long)  

    
def enhance_TUDataset(ds, node_feature):
    
    assert node_feature in ['original', 'node_degree', 'one_hot_node_degree']
    
    X = []
    targets = []
    
    max_degree_by_graph = []
    num_nodes = []
    num_edges = []
    
    for x in ds:
        
        targets.append(x.y.item())
        
        boundary_info = get_boundary_info(x)
        x.boundary_info = boundary_info
        
        num_nodes.append(x.num_nodes)
        num_edges.append(boundary_info.size(0))
        
        degree = torch.zeros(x.num_nodes, dtype=torch.long)

        for k, v in Counter(x.boundary_info.flatten().tolist()).items():
            degree[k] = v
        max_degree_by_graph.append(degree.max().item())
            
        x.node_deg = degree
        X.append(x)
        
    max_node_deg =  max(max_degree_by_graph)
    
    if node_feature != 'original':
        for d in X:

            if node_feature == 'node_degree':
                d.x = d.node_deg
                
            elif node_feature == 'one_hot_node_degree':
                degree_ohe = torch.zeros(d.num_nodes, max_node_deg + 1, dtype=torch.float)
                degree_ohe.scatter_(1, d.node_deg.unsqueeze(1), 1)
                d.x = degree_ohe
    
    feature_dim = 1 if X[0].x.dim() == 1 else X[0].x.size()[1]
    if node_feature not in ['node_degree']:        
        assert all((feature_dim == d.x.size()[1] for d in X))
                
    new_ds =  SimpleDataset(X)
    
    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes   = len(set(targets))
    new_ds.feature_dim  = feature_dim
    
    return new_ds


def dataset_factory(dataset_name, node_feature):
    if dataset_name in TU_DORTMUND_DATASET_NAMES:

        path = '/home/pma/chofer/repositories/pytorch_geometric/data/{}/'.format(dataset_name)
        dataset = TUDataset(path, name=dataset_name)

    elif dataset_name in POWERFUL_GNN_DATASET_NAMES:

        dataset = load_powerfull_gnn_dataset(dataset_name)

    else:
        raise ValueError("dataset_name not in {}".format(TU_DORTMUND_DATASET_NAMES + POWERFUL_GNN_DATASET_NAMES))

    print("# Dataset: ", dataset.name)

    dataset = enhance_TUDataset(dataset, node_feature=node_feature)
    
    print('# num samples: ', len(dataset))
    print('# num classes: ', dataset.num_classes)
    print('#')
    print('# max vertex degree: ', dataset.max_node_deg)
    print('# avg number of nodes: ', dataset.avg_num_nodes)
    print('# avg number of edges: ', dataset.avg_num_edges)
    print('# feature_dim: ', dataset.feature_dim)
    
    return dataset


def train_test_split(dataset, seed=0, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = seed)

    targets = [x.y.item() for x in dataset]
    split_idx = list(skf.split(np.zeros(len(dataset)), targets))
    print('# num splits: ', len(split_idx))

    ret = []
    for train_i, test_i in split_idx:
        train = Subset(dataset=dataset, indices=train_i)
        test  = Subset(dataset=dataset, indices=test_i)
        
        ret.append((train, test))

    return ret


# def forward(fil, cls, batch, device):

#     node_feat   = batch.x.to(device)
#     edge_index = batch.edge_index.to(device)

#     vertex_filtration = fil(node_feat, edge_index)

#     ph_input = []
#     for i, j, e in zip(batch.sample_pos[:-1],batch.sample_pos[1:], batch.boundary_info):
#         v = vertex_filtration[i:j,:]
#         ph_input.append((v.squeeze().to(device), [e.to(device)]))

#     pers = ph(ph_input)

#     h0 = [x[0][0] for x in pers]
#     h0ess = [x[1][0].unsqueeze(1) for x in pers]
#     h1ess = [x[1][1].unsqueeze(1) for x in pers]

#     y_hat = cls(h0, h0ess, h1ess)
    
#     return y_hat


def evaluate(dataloader, models, device):
    num_samples = 0
    correct = 0 

    models = list(models)
    
    models = [m.eval().to(device) for m in models]

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch.boundary_info = [e.to(device) for e in batch.boundary_info]
            
            y_hat      = sum((m(batch) for m in models))
            
            y_pred = y_hat.max(dim=1)[1]    
            
            correct += (y_pred == batch.y).sum().item()
            num_samples += batch.y.size(0)
        
    return float(correct)/ float(num_samples) 


import torch.nn as nn
class DeepSets(nn.Module):    
    aggregation_f = {
            'add': torch.sum, 
            'mean': torch.mean, 
            'max' : lambda *args, **kwargs: torch.max(*args, **kwargs)[0], 
            'min' : lambda *args, **kwargs: torch.min(*args, **kwargs)[0]
    }
    
    def __init__(self, feat_out, point_dim, aggr='add'):
        super().__init__()
        
        self.aggregate = self.aggregation_f[aggr]       
        
        self.phi = nn.Linear(point_dim, feat_out)
        
    def forward(self, batch):
        slice_i = [0] + [t.size(0) for t in batch]
        slice_i = torch.tensor(slice_i).cumsum(0)  
        
        x = torch.cat(batch, dim=0)
        x = self.phi(x)
        
        tmp = []
        for i, j in zip(slice_i[:-1], slice_i[1:]):
            tmp.append(self.aggregate(x[i:j], dim=0))
            
        x = torch.stack(tmp, 0)
        
        return x