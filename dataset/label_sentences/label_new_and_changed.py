import torch
import json
from simcse import SimCSE
from tools import Classifier
from pathlib import Path
import argparse
import os

from label_not_same import save_json, save_and_load_json, init
from fairseq.data.data_utils import collate_tokens

torch.manual_seed(0)
print('Set random seed as 0')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--src_root', type=str, default='label_result_not_same')
    parser.add_argument('--save_root', type=str, default='label_result_new_and_changed')
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

    
def compare_article(sim_model, clf_model, js_old, js_new):
    
    # label C, N
    classifier = Classifier(sim_model, clf_model)
    result_old, result_new = classifier.label_changed_and_new_sentences(js_old['NS'], js_new['NS'])
    
    # update same
    if 'S' in js_old and 'indices' in js_old['S']:
        js_old['S'] = list_to_dict(js_old['S']) 

    # update C
    js_old['C'] = result_old['C']

    # update N
    if 'N' in js_new and 'indices' in js_new['N']:
        js_new['N']['indices']= [i for lst in js_new['N']['indices'] for i in lst]
        js_new['N']['indices'].extend(result_new['N']['indices'])
        js_new['N']['sentences'].extend(result_new['N']['sentences'])
    else:
        js_new['N'] = result_new['N']
    
    del js_old['NS']
    del js_new['NS'], js_new['S']
    
    return js_old, js_new


def run(args):
    
    month_old = args.month_old
    month_new = month_old + 1
    src_root = args.src_root + f'/{month_old:02d}{month_new:02d}'
    save_root = args.save_root + f'/{month_old:02d}{month_new:02d}'
    sim_model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    clf_model = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').eval()
    
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
            result_old, result_new = compare_article(sim_model, clf_model, js_old[str(curr_idx)], js_new[str(curr_idx)])
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
    run(args)
    
