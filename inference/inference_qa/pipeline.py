from retriever import retrieve, reretrieve
from classification.model import get_model
from classification.dataset import FilterDataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from dataset import RetrievalDataset
import os
import json
import fire
from tqdm import tqdm

def main(month: int = 9,
         data_root: str = 'QA_dataset',
         llama_config_dir: str = 'meta-llama/Llama-2-7B',
         llama_ckpt_path: str = None,
         pred_ckpt_path: str = 'classification/ckpt',
         batch_size: int = 8,
         save_root: str = 'results',
         db_faiss_dir: str = 'vectorstore/sentbert',
         model_name: str = None,
         hp_lambda = 2.
        ):
    
    if model_name is None:
        if 'sentbert' in db_faiss_dir:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        elif 'contriever' in db_faiss_dir:
            model_name =D
        if 'e5' in db_faiss_dir:
            model_name = 'intfloat/e5-large-v2'
    db_type = db_faiss_dir.split('/')[-1]
    
    # Init model
    accelerator = Accelerator()
    
    dataset = RetrievalDataset(month, data_root)    
            
    # Retrieve
    retrievals = retrieve(accelerator,
                          month,
                          dataset,
                          return_results=True,
                          save=True,
                          db_faiss_dir=db_faiss_dir,
                          model_name=model_name
                          )
    
    # Prepare model and dataset
    tokenizer, model = get_model(accelerator, 
                                 llama_config_dir, 
                                 llama_ckpt_path, 
                                 train=False, 
                                 pred_ckpt_path=pred_ckpt_path
                                 )
    accelerator.wait_for_everyone()
    model.eval()
    
    dataset = FilterDataset(dataset=retrievals, mode='test', concat=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    dataloader = accelerator.prepare(dataloader)
    
    # First inference
    all = []
    for batch in tqdm(dataloader):
        qaids, prompts, metas = batch
        outputs = model.inference(prompts)
        for qaid, out, meta in zip(qaids, outputs, metas):
            meta['prediction'] = out # [prob_for_label_0, label, generated_text]
            label = out['label']
            all.append([qaid, meta])
    all = gather_object(all)
    all = sorted(all, key=lambda x: int(x[0]))

    save_dir = f'{save_root}_{db_type}/{month:02d}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_root}_{db_type}/{month:02d}/all.json', 'w') as f:
        json.dump(all, f, indent = 2)
        
    # Decision Gate
    final_output = {}
    to_reretrieve = {}
    for qid, qitem in all:
        if qid not in final_output:
            final_output[qid] = qitem
        else:
            if final_output[qid]['prediction']['prob'] < qitem['prediction']['prob']:
                final_output[qid] = qitem 
    
    for qid, qitem in final_output.items():
        if qitem['prediction']['label'] > 0:
            to_reretrieve[qid] = qitem
            
    with open(f'{save_root}_{db_type}/{month:02d}/selected.json', 'w') as f:
        json.dump(final_output, f, indent = 2)
    with open(f'{save_root}_{db_type}/{month:02d}/reretrieve.json', 'w') as f:
        json.dump(to_reretrieve, f, indent = 2)


    # Adaptive Re-retrieval
    dataset = RetrievalDataset(month, dataset=to_reretrieve)
    retrievals = reretrieve(accelerator,
                          month,
                          dataset,
                          return_results=True,
                          save=False,
                          db_faiss_dir=db_faiss_dir,
                          model_name=model_name,
                          hp_lambda=hp_lambda
                          )
    
    # Dataset for second inference
    dataset = FilterDataset(dataset=retrievals, mode='test', retrieval_key='reretrieval')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    dataloader = accelerator.prepare(dataloader)
    
    # Second inference
    all = []
    for batch in tqdm(dataloader):
        qaids, prompts, metas = batch
        outputs = model.inference(prompts)
        for qaid, out, meta in zip(qaids, outputs, metas):
            old_prob, new_prob = meta["prediction"]["prob"], out["prob"]
            meta['old_prediction'] = meta['prediction']
            meta['new_prediction'] = out
            meta['prediction'] = meta['old_prediction'] if old_prob > new_prob else out 
            all.append([qaid, meta])
    all = gather_object(all)
        
    # Select final label    
    for qid, qitem in all:
        if qid not in final_output:
            final_output[qid] = qitem
        else:
            if final_output[qid]['prediction']['prob'] < qitem['prediction']['prob']:
                final_output[qid] = qitem 
    
    with open(f'{save_root}_{db_type}/{month:02d}/final_output.json', 'w') as f:
        json.dump(final_output, f, indent = 2)

    
if __name__ == "__main__":
    fire.Fire(main)