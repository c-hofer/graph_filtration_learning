import json
import argparse

from core.train_engine import experiment_multi_device


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)


    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cfg_file', type=str, 
                        help='')
    parser.add_argument('--output_dir', type=str, 
                        help='')
    parser.add_argument('--devices', nargs='+')

    parser.add_argument('--max_process_on_device', type=int)


    args = parser.parse_args()

    devices = [int(d) for d in args.devices]

    with open(args.cfg_file, 'r') as fid:
        exp_cfgs = json.load(fid)
        
    experiment_multi_device(exp_cfgs, args.output_dir, devices, args.max_process_on_device)


