import torch
import json
from simcse import SimCSE
from tools import Comparator
from pathlib import Path
import argparse
import os
        
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_old', type=int, default=8)
    parser.add_argument('--save_root', type=str, default='label_result_NS')
    parser.add_argument('--root', type=str, default = 'TemporalWikiDatasets/Wikipedia_datasets')
    args = parser.parse_args()
    return args


def match_article(model, old_text, new_text, thrs = 0.99):     
    
    comp = Comparator(model, old_text, new_text)
    comp.label_same(thrs)
    result_old, result_new = comp.get_label_same_result()
    
    return result_old, result_new
        

def save_json(save_root, month, file_list, results, new=True):
    # set dir
    save_dir = f"{save_root}/{month:02d}/{file_list[0].split('/')[-2]}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # save json
    save_fn = save_dir + "/" + file_list[0].split('/')[-1]
    with open(save_fn, 'w') as f:
        json.dump(results, f, indent=2)



def save_and_load_json(save_root, month, file_list, results, new=True):    
    
    save_json(save_root, month, file_list, results, new=True)
    if new:
        print('Dump new :', '/'.join(file_list[0].split('/')[-2:]))
    else:
        print('Dump old :', '/'.join(file_list[0].split('/')[-2:]))
    
    if len(file_list) == 1:
        return None, None, None

    del file_list[0]
    with open(file_list[0]) as f:
        js = json.load(f)
    max_idx = max([int(i) for i in js])    
    
    return file_list, js, max_idx
    
    

def init(root, save_root, month, new=True):
    
    result_files = [str(i) for i in sorted(Path(save_root).glob(f'{month:02}/*/wiki_*.json'))]
    js_files = [str(i) for i in sorted(Path(root).glob(f'{month:02}/*/wiki_*.json')) if not str(i).endswith('result.json')]
    
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
    root = args.root
    save_root = args.save_root + f'/{month_old:02d}{month_new:02d}'
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large', batch_size=1024)
    
    results_old, files_old, js_old = init(root, save_root, month_old, False)
    results_new, files_new, js_new = init(root, save_root, month_new)
    
    curr_idx = min([int(i) for i in js_new])
    max_idx_old = max([int(i) for i in js_old])
    max_idx_new = max([int(i) for i in js_new])
    
    print('Start from', files_new[0])
    print('Start index is', curr_idx)
    
    while True:
        
        # if old js ended, save and load next old js
        if curr_idx > max_idx_old:
            files_old, js_old, max_idx_old = save_and_load_json(save_root, month_old, files_old, results_old, new=False)
            results_old = {}
        
        # no key in old
        if str(curr_idx) not in js_old:
            if len(js_new[str(curr_idx)][2]) > 0:
                results_new[str(curr_idx)] = 'N'
        
        # no text in old
        elif len(js_old[str(curr_idx)][2]) == 0:
            if len(js_new[str(curr_idx)][2]) > 0:
                results_new[str(curr_idx)] = 'N' 
            else:
                pass
        
        # text in old
        else:
            # no text in new 
            if len(js_new[str(curr_idx)][2]) == 0:
                results_old[str(curr_idx)] = 'D'
            # text in both old and new -> compare
            else:
                article_old = js_old[str(curr_idx)][2]
                article_new = js_new[str(curr_idx)][2]
                
                result_old, result_new = match_article(model, article_old, article_new)
                results_old[str(curr_idx)] = result_old
                results_new[str(curr_idx)] = result_new
                
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
    