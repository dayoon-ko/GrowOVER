import torch
import json
from simcse import SimCSE
from tools import Prompter
from pathlib import Path
import argparse
import os
from label_NS import save_json, save_and_load_json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--src_root', type=str, default='label_result_NS')
    parser.add_argument('--save_root', type=str, default='label_result_N_C')
    args = parser.parse_args()
    return args


def list_to_dict(res):
    new_s = {}
    for old_idx, new_idx in zip(res['indices'], res['match_indices']):
        if type(new_idx) == int:
            new_s[old_idx] = new_idx
        elif type(new_idx) == list:
            # select the list with minimum length
            min_len, min_idx = new_idx[0][1] - new_idx[0][0], 0
            for i, (s, e) in enumerate(new_idx):
                if e-s < min_len:
                    min_len = e-s
                    min_idx = i
            new_s[old_idx] = new_idx[min_idx]
    return new_s

    
def compare_article(model, js_old, js_new):
    
    # label C, D, N
    prompter = Prompter(model)
    result_old, result_new = prompter.label_changed_and_new_sentences(js_old['NS'], js_new['NS'])
    
    # update same
    if 'S' in js_old and 'indices' in js_old['S']:
        js_old['S'] = list_to_dict(js_old['S']) 
    
    # update C
    js_old['C'] = result_old['C']
    
    # update N
    if 'N' in js_new and 'indices' in js_new['N']:
        js_new['N'] = {i:None for lst in js_new['N']['indices'] for i in lst}
    js_new['N'].update(result_new['N'])
    
    # update S2
    js_old['S2'] = result_old['S2']    
    
    del js_old['NS']
    del js_new['NS'], js_new['S']
    
    return js_old, js_new


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
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    
    results_old, files_old, js_old = init(src_root, save_root, month_old)
    results_new, files_new, js_new = init(src_root, save_root, month_new)
    
    curr_idx = min([int(i) for i in js_new])
    max_idx_old = max([int(i) for i in js_old])
    max_idx_new = max([int(i) for i in js_new])
    
    print('Start from', files_old[0])
    print('Start index is', curr_idx)
    
    while True:
        
        # if old js ended, save and load next old js
        if curr_idx > max_idx_old:
            # save
            files_old, js_old, max_idx_old = save_and_load_json(save_root, month_old, files_old, results_old, new=False)
            results_old = {}
        
        # New
        if str(curr_idx) not in js_old:
            results_new[str(curr_idx)] = js_new[str(curr_idx)]
        
        else:
            print('\n' + str(curr_idx))
            result_old, result_new = compare_article(model, js_old[str(curr_idx)], js_new[str(curr_idx)])
            results_old[str(curr_idx)] = result_old
            results_new[str(curr_idx)] = result_new
            print('------------------')
            
            save_json(save_root, month_old, files_old, results_old, new=False)
            save_json(save_root, month_new, files_new, results_new, new=True)            
            
        del js_new[str(curr_idx)]
        
        # if new js ended, save and load next new js
        if len(js_new) == 0:
            files_new, js_new, max_idx_new = save_and_load_json(save_root, month_new, files_new, results_new) 
            results_new = {}
            # dump old js before break, if ended
            if files_new is None:
                save_json(save_root, month_old, files_old, results_old, new=False)
                break
        
        # update curr_idx
        curr_idx = min([int(i) for i in js_new]) 

    
    
if __name__ == "__main__":
    args = get_args()
    main(args)