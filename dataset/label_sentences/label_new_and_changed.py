import torch
import json
from simcse import SimCSE
from tools import Classifier
import argparse
import os
from tools import list_to_dict
from label_not_same import save_json, save_and_load_json, init

torch.manual_seed(0)
print('Set random seed as 0')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--root', type=str, default='label_result_not_same')
    parser.add_argument('--save_root', type=str, default='label_result_new_and_changed')
    args = parser.parse_args()
    return args


def run(args):
    
    month_old = args.month_old
    month_new = month_old + 1
    root = args.root + f'/{month_old:02d}{month_new:02d}'
    save_root = args.save_root + f'/{month_old:02d}{month_new:02d}'
    sim_model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    clf_model = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').eval()
    classifier = Classifier(sim_model, clf_model)
    
    results_old, input_files_old, input_old = init(root, save_root, month_old)
    results_new, input_files_new, input_new = init(root, save_root, month_new)
    
    curr_idx = min([int(i) for i in input_new])
    max_idx_old = max([int(i) for i in input_old])
    max_idx_new = max([int(i) for i in input_new])
    
    print('Start from', input_files_old[0])
    print('Start index is', curr_idx)
    
    # iterate over each article (curr_idx) across wiki_*.json files, 
    # finding new and changed sentences in each old and new article
    while True:
        
        # if input_old ended, save results_old and load next input_old
        if curr_idx > max_idx_old:
            # save
            input_files_old, input_old, max_idx_old = save_and_load_json(save_root, month_old, input_files_old, results_old, new=False)
            results_old = {}
        
        # if curr_idx is only in input_new, "New"
        if str(curr_idx) not in input_old:
            results_new[str(curr_idx)] = input_new[str(curr_idx)]
        
        # otherwise, compare old and new articles
        else:
            result_old, result_new = classifier.label_changed_and_new_sentences(
                                        input_old[str(curr_idx)], 
                                        input_new[str(curr_idx)]
                                    )
            results_old[str(curr_idx)] = result_old
            results_new[str(curr_idx)] = result_new
            
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
    
