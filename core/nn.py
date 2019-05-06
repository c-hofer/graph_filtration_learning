import torch
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
