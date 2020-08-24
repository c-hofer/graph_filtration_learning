import os
import pickle
import glob


def read_exp_result_files(path):
    files = glob.glob(os.path.join(path, "*.pickle"))
    res = []
    for f in files: 
        if os.path.basename(f) == 'errors.pickle':
            continue        
        
        r = pickle.load(open(f, 'rb'))
        
        #older cfgs have no 'set_node_degree_uninformative' ... 
        if 'set_node_degree_uninformative' not in r['exp_cfg']['model']:
            r['exp_cfg']['model']['set_node_degree_uninformative'] = False
            
        res.append(r)
    return res