import torch
import json
from simcse import SimCSE
from tools import Filter
from pathlib import Path
import argparse
import os
from tools import save_json, save_and_load_json, init

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--root', type=str, default='label_result_N_C')
    parser.add_argument('--save_root', type=str, default='filter_result_C')
    args = parser.parse_args()
    return args

def filter_ctd(filter, input_old):    
    # filter changed
    res = filter.run_filter(input_old['C'])
    input_old['C'] = res
    return input_old

def run(run):   
    month_old = args.month_old
    month_new = month_old + 1
    root = args.root + f'/{month_old:02d}{month_new:02d}'
    save_root = args.save_root + f'/{month_old:02d}{month_new:02d}'
    sim_model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    filter = Filter(sim_model)
    
    # initialize old inputs and results
    results_old, input_files_old, input_old = init(root, save_root, month_old)
    
    # set starting article id
    next_indices = [int(i) for i in input_old]
    if len(next_indices) > 0:
        curr_idx = min(next_indices)
    else:
        with open(input_files_old[0]) as f:
            input_old = json.load(f)
            results_old = {}
            curr_idx = min([int(i) for i in input_old])
    print('Start from', input_files_old[0])
    print('Start index is', curr_idx)
    
    # iterate over each article (curr_idx) across wiki_*.json files, 
    # filtering changed sentence pairs 
    while True:
        if len(input_old[str(curr_idx)]['C']['indices']) > 0:
            result_old = filter_ctd(filter, input_old[str(curr_idx)])
            save_json(save_root, month_old, input_files_old, results_old, new=False) 
        else:
            result_old = input_old[str(curr_idx)]
        results_old[str(curr_idx)] = result_old    
            
        del input_old[str(curr_idx)]
        
        # if old js ended, save and load next old js
        if len(input_old) == 0:
            input_files_old, input_old, max_idx_old = save_and_load_json(save_root, month_old, input_files_old, results_old) 
            results_old = {}
            # break, if ended
            if input_files_old is None:
                break
        
        # update curr_idx
        curr_idx = min([int(i) for i in input_old]) 

    
if __name__ == "__main__":
    args = get_args()
    run(args)
    
