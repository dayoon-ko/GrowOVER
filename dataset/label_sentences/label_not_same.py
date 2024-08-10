import torch
import json
from simcse import SimCSE
from tools import (
    Comparator, 
    load_json, 
    match_article, 
    save_json, 
    save_and_load_json, 
    init
)
from pathlib import Path
import argparse
import os

        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--save_root', type=str, default='label_result_not_same')
    parser.add_argument('--root', type=str, default = '')
    args = parser.parse_args()
    return args
    


def run(args):
    
    month_old = args.month_old
    month_new = month_old + 1
    root = args.root
    save_root = args.save_root + f'/{month_old:02d}{month_new:02d}'
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large', batch_size=1024)
    
    # initilaize results, files, and input json files
    results_old, input_files_old, input_old = init(root, save_root, month_old, False)
    results_new, input_files_new, input_new = init(root, save_root, month_new)
    
    curr_idx = min([int(i) for i in input_new])
    max_idx_old = max([int(i) for i in input_old])
    max_idx_new = max([int(i) for i in input_new])
    
    print('Start from', input_files_new[0])
    print('Start index is', curr_idx)
    
    # iterate over each article (curr_idx) across wiki_*.json files, 
    # matching each old and new article
    while True:

        # if input_old ended, save results_old and load next input_old
        if curr_idx > max_idx_old:
            input_files_old, input_old, max_idx_old = save_and_load_json(save_root, month_old, input_files_old, results_old, new=False)
            results_old = {}
        
        # input_old has no key 
        if str(curr_idx) not in input_old:
            if len(input_new[str(curr_idx)]["text"]) > 0:
                results_new[str(curr_idx)] = 'N'
        
        # input_old has no text
        elif len(input_old[str(curr_idx)]["text"]) == 0:
            if len(input_new[str(curr_idx)]["text"]) > 0:
                results_new[str(curr_idx)] = 'N' 
            else:
                pass
        
        # input_old has text
        else:
            # input_new has no text
            if len(input_new[str(curr_idx)]["text"]) == 0:
                results_old[str(curr_idx)] = 'D'
            # input_new has text, then compare
            else:
                article_old = input_old[str(curr_idx)]["text"]
                article_new = input_new[str(curr_idx)]["text"]
                
                result_old, result_new = match_article(model, article_old, article_new)
                if result_old is not None:
                    results_old[str(curr_idx)] = result_old
                    results_new[str(curr_idx)] = result_new
                    
                    # save intermediate results
                    save_json(save_root, month_old, input_files_old, results_old, new=False)
                    save_json(save_root, month_new, input_files_new, results_new, new=True)
        
        # remove the input done         
        del input_new[str(curr_idx)]
        
        # if input_new ends, save input_new and load next input_new
        if len(input_new) == 0:
            input_files_new, input_new, max_idx_new = save_and_load_json(save_root, month_new, input_files_new, results_new)
            results_new = {}
            # dump input_old before break, if ended
            if input_files_new is None:
                save_json(save_root, month_old, input_files_old, results_old, new=False)
                break
        
        # update curr_idx
        curr_idx = min([int(i) for i in input_new]) 
            

        
if __name__ == "__main__":
    args = get_args()
    run(args)
    