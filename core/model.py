import torch
import numpy as np
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geonn

import functools

from chofer_torchex.nn import SLayerRationalHat
from torch_geometric.nn import GINConv, global_add_pool, global_sort_pool


from chofer_torchex import pershom
ph = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch


def gin_mlp_factory(gin_mlp_type: str, dim_in: int, dim_out: int):
    if gin_mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif gin_mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in), 
            nn.LeakyReLU(), 
            nn.Linear(dim_in, dim_out)
        )

    elif gin_mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in), 
            nn.BatchNorm1d(dim_in), 
            nn.LeakyReLU(), 
            nn.Linear(dim_in, dim_out)
        )
    else: 
        raise ValueError("Unknown gin_mlp_type!")


def ClassifierHead(
    dataset, 
    dim_in: int=None, 
    hidden_dim: int=None, 
    drop_out: float=None):

    assert (0.0 <= drop_out) and (drop_out < 1.0)
    assert dim_in is not None
    assert drop_out is not None
    assert hidden_dim is not None

    tmp = [
        nn.Linear(dim_in, hidden_dim), 
        nn.LeakyReLU(),             
    ]

    if drop_out > 0:
        tmp += [nn.Dropout(p=drop_out)]

    tmp += [nn.Linear(hidden_dim, dataset.num_classes)]

    return nn.Sequential(*tmp)


class DegreeOnlyFiltration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch):
        tmp = []
        for i, j in zip(batch.sample_pos[:-1], batch.sample_pos[1:]):
            max_deg = batch.node_deg[i:j].max()
            
            t = torch.ones(j - i, dtype=torch.float, device=batch.node_deg.device)
            t = t * max_deg 
            tmp.append(t)

        max_deg = torch.cat(tmp, dim=0)
                    
        normalized_node_deg = batch.node_deg.float() / max_deg

        return normalized_node_deg 


class Filtration(torch.nn.Module):
    def __init__(self, 
            dataset,  
            use_node_degree=None,
            set_node_degree_uninformative=None,
            use_node_label=None, 
            gin_number=None,
            gin_dimension=None,
            gin_mlp_type = None, 
            **kwargs
        ):
        super().__init__()
        
        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = nn.Embedding(max_node_deg+1, dim)
        else:
            self.embed_deg = None

        self.embed_lab = nn.Embedding(num_node_lab, dim) if use_node_label else None
        
        dim_input = dim*((self.embed_deg is not None) + (self.embed_lab is not None))  
        
        dims = [dim_input] + (gin_number)*[dim]
        
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
            self.convs.append(GINConv(l, train_eps=True))
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
    def __init__(self, 
            dataset, 
            num_struct_elements=None, 
            cls_hidden_dimension=None, 
            drop_out=None, 
        ):

        super().__init__()
        assert isinstance(num_struct_elements, int) 
        self.use_as_feature_extractor = False
        
        self.ldgm_0     = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1) 
        self.ldgm_0_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        self.ldgm_1_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1) 
        fc_in_feat = 3*num_struct_elements
        
        self.cls_head = ClassifierHead(
            dataset, 
            dim_in=fc_in_feat, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )      
                        
    def forward(self, h_0, h_0_ess, h_1_ess):
        tmp = []
        
        tmp.append(self.ldgm_0(h_0))            
        tmp.append(self.ldgm_0_ess(h_0_ess))            
        tmp.append(self.ldgm_1_ess(h_1_ess))
            
        x = torch.cat(tmp, dim=1)

        if not self.use_as_feature_extractor:        
            x = self.cls_head(x)
        
        return x


class PershomBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.use_super_level_set_filtration = None
        self.use_as_feature_extractor = False
        self.fil = None
        self.cls = None

    def forward(self, batch):    
        assert self.use_super_level_set_filtration is not None
        
        node_filt = self.fil(batch) 
       
        ph_input = []
        for i, j, e in zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_info):
            v = node_filt[i:j]
            ph_input.append((v, [e]))

        pers = ph(ph_input)

        if not self.use_super_level_set_filtration:
            h_0 = [x[0][0] for x in pers]
            h_0_ess = [x[1][0].unsqueeze(1) for x in pers]
            h_1_ess = [x[1][1].unsqueeze(1) for x in pers]

        else:
            ph_sup_input = [(-v, e) for v, e in ph_input]
            pers_sup = ph(ph_sup_input)

            h_0 = [torch.cat([x[0][0], -(y[0][0])], dim=0) for x, y in zip(pers, pers_sup)]
            h_0_ess = [torch.cat([x[1][0], -(y[1][0])], dim=0).unsqueeze(1) for x, y in zip(pers, pers_sup)]
            h_1_ess = [torch.cat([x[1][1], -(y[1][1])], dim=0).unsqueeze(1) for x, y in zip(pers, pers_sup)]       

        y_hat = self.cls(h_0, h_0_ess, h_1_ess)

        return y_hat  

    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    @property
    def use_as_feature_extractor(self):
        return self.use_as_feature_extractor

    @use_as_feature_extractor.setter
    def use_as_feature_extractor(self, val):
        if hasattr(self, 'cls'):
            self.cls.use_as_feature_extractor = val

    
    def init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.apply(init)

    
class PershomLearnedFilt(PershomBase):
    def __init__(self,
                dataset,
                use_super_level_set_filtration: bool=None, 
                use_node_degree: bool=None, 
                set_node_degree_uninformative: bool=None, 
                use_node_label: bool=None, 
                gin_number: int=None, 
                gin_dimension: int=None,
                gin_mlp_type: str=None, 
                num_struct_elements: int=None,
                cls_hidden_dimension: int=None,
                drop_out: float=None,
                **kwargs,                 
                ):
        super().__init__()


        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = Filtration(
            dataset,
            use_node_degree=use_node_degree,
            set_node_degree_uninformative=set_node_degree_uninformative, 
            use_node_label=use_node_label, 
            gin_number=gin_number,
            gin_dimension=gin_dimension,
            gin_mlp_type =gin_mlp_type,  
        )

        self.cls = PershomClassifier(
            dataset,
            num_struct_elements=num_struct_elements,
            cls_hidden_dimension=cls_hidden_dimension,
            drop_out=drop_out
        )

        self.init_weights()   


class PershomRigidDegreeFilt(PershomBase):
    def __init__(self,
                dataset,
                use_super_level_set_filtration: bool=None, 
                num_struct_elements: int=None,
                cls_hidden_dimension: int=None,   
                drop_out: float=None,   
                **kwargs,            
                ):
        super().__init__()


        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = DegreeOnlyFiltration()

        self.cls = PershomClassifier(
            dataset,
            num_struct_elements=num_struct_elements, 
            drop_out=drop_out, 
            cls_hidden_dimension=cls_hidden_dimension
        )

        self.init_weights() 


class OneHotEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        eye = torch.eye(dim, dtype=torch.float)

        self.register_buffer('eye', eye)

    def forward(self, batch):
        assert batch.dtype == torch.long 

        return self.eye.index_select(0, batch)
    
    @property
    def dim(self):
        return self.eye.size(1)


class UniformativeDummyEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        b = torch.ones(1, dim, dtype=torch.float)
        self.register_buffer('ones', b)

    def forward(self, batch):
        assert batch.dtype == torch.long 
        return self.ones.expand(batch.size(0), -1)
    
    @property
    def dim(self):
        return self.ones.size(1)


class GIN(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))   
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError

        self.cls = ClassifierHead(
            dataset, 
            dim_in=gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

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
  
        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()

        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x

