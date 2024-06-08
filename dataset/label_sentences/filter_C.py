import torch
import json
from simcse import SimCSE
from tools import Filter
from pathlib import Path
import argparse
import os
from label_NS import save_json, save_and_load_json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--src_root', type=str, default='label_result_N_C')
    parser.add_argument('--save_root', type=str, default='filter_result_C')
    args = parser.parse_args()
    return args

    
def filter_ctd(filter, js_old):    
    # filter C
    res = filter.run_filter(js_old['C'])
    js_old['C'] = res
    return js_old


def init(root, save_root, month, new=True):
    result_files = [str(i) for i in sorted(Path(save_root).glob(f'{month:02}/*/wiki_*.json'))]
    js_files = [str(i) for i in sorted(Path(root).glob(f'{month:02}/*/wiki_*.json'))]
  
    if len(result_files) == 0:
        with open(js_files[0]) as f:
            js = json.load(f)
        return {}, js_files, js
    
    else:
        start_idx = len(result_files) - 1
        js_files = js_files[start_idx:]
        with open(js_files[0]) as f:
            js = json.load(f)
        with open(result_files[-1]) as f:
            result_js = json.load(f)
        if new:
            for k in result_js:
                del js[k]
        return result_js, js_files, js



def main(args):
    
    month_old = args.month_old
    month_new = month_old + 1
    src_root = args.src_root + f'/{month_old:02d}{month_new:02d}'
    save_root = args.save_root + f'/{month_old:02d}{month_new:02d}'
    sim_model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    filter = Filter(sim_model)
    
    results_old, files_old, js_old = init(src_root, save_root, month_old)
    
    next_indices = [int(i) for i in js_old]
    if len(next_indices) > 0:
        curr_idx = min(next_indices)
    else:
        with open(files_old[0]) as f:
            js_old = json.load(f)
            results_old = {}
            curr_idx = min([int(i) for i in js_old])
    
    print('Start from', files_old[0])
    print('Start index is', curr_idx)
    
    while True:
        
        if len(js_old[str(curr_idx)]['C']['indices']) > 0:
            print('\n' + str(curr_idx))
            result_old = filter_ctd(filter, js_old[str(curr_idx)])
            save_json(save_root, month_old, files_old, results_old, new=False) 
            print('------------------')
        else:
            result_old = js_old[str(curr_idx)]
        results_old[str(curr_idx)] = result_old    
            
        del js_old[str(curr_idx)]
        
        # if old js ended, save and load next old js
        if len(js_old) == 0:
            files_old, js_old, max_idx_old = save_and_load_json(save_root, month_old, files_old, results_old) 
            results_old = {}
            # break, if ended
            if files_old is None:
                break
        
        # update curr_idx
        curr_idx = min([int(i) for i in js_old]) 

    
    
if __name__ == "__main__":
    args = get_args()
    main(args)
    
