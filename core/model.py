import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geonn

from chofer_torchex.nn import SLayerRationalHat
from torch_geometric.nn import GINConv, global_add_pool


from chofer_torchex import pershom
ph = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch


class DegreeOnlyFiltration(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, batch):
        return batch.node_deg.float()


class Filtration(torch.nn.Module):
    def __init__(self, 
            dataset,  
            use_node_deg=None,
            use_node_lab=None, 
            num_gin=1,
            hidden_dim=32,
            use_mlp = False
        ):
        super().__init__()
        
        dim = hidden_dim
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        self.embed_deg = nn.Embedding(max_node_deg+1, dim) if use_node_deg else None
        self.embed_lab = nn.Embedding(num_node_lab, dim) if use_node_lab else None
        
        dim_input = dim*((self.embed_deg is not None) + (self.embed_lab is not None))  
        
        dims = [dim_input] + (num_gin)*[dim]
        
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = nn.Linear(n_1, n_2) if not use_mlp else nn.Sequential(nn.Linear(n_1, n_1), nn.LeakyReLU(), nn.Linear(n_1, n_2))
            self.convs.append(GINConv(l))
            self.bns.append(nn.BatchNorm1d(n_2))             

        self.fc = nn.Sequential(
            nn.Linear(sum(dims), dim),
            nn.BatchNorm1d(dim), 
            nn.LeakyReLU(),
            nn.Linear(dim, 1), 
            nn.Sigmoid()
        )

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        x = torch.cat(z, dim=1)
        ret = self.fc(x).squeeze()
        return ret


class PershomClassifier(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        
        self.ldgm_0     = SLayerRationalHat(50, 2, radius_init=0.1) 
        self.ldgm_0_ess = SLayerRationalHat(50, 1, radius_init=0.1)
        self.ldgm_1_ess = SLayerRationalHat(50, 1, radius_init=0.1) 
        fc_in_feat = 150
        
        self.fc = nn.Sequential(
            nn.Linear(fc_in_feat, 100), 
            nn.BatchNorm1d(100),
            nn.LeakyReLU(), 
            nn.Linear(100, dataset.num_classes)
        )        
                        
    def forward(self, h_0, h_0_ess, h_1_ess):
        tmp = []
        
        tmp.append(self.ldgm_0(h_0))            
        tmp.append(self.ldgm_0_ess(h_0_ess))            
        tmp.append(self.ldgm_1_ess(h_1_ess))
            
        x = torch.cat(tmp, dim=1)        
        x = self.fc(x)
        
        return x
    
    
class PershomModel(nn.Module):
    def __init__(self,
                dataset,
                filtration_kwargs=None,                 
                ):
        super().__init__()
        self.fil = Filtration(dataset, **filtration_kwargs) if filtration_kwargs is not None else DegreeOnlyFiltration()       
        self.cls = PershomClassifier(dataset)
        self.init_weights()
        
        
    def forward(self, batch):    
        
        node_filt = self.fil(batch) 
       
        ph_input = []
        for i, j, e in zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_info):
            v = node_filt[i:j]
            ph_input.append((v, [e]))

        pers = ph(ph_input)
            

        h_0 = [x[0][0] for x in pers]
        h_0_ess = [x[1][0].unsqueeze(1) for x in pers]
        h_1_ess = [x[1][1].unsqueeze(1) for x in pers]

        y_hat = self.cls(h_0, h_0_ess, h_1_ess)

        return y_hat
    
    
    def init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.apply(init)
    
