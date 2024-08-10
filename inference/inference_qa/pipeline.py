from retriever import retrieve
from classification.model import get_model
from classification.dataset import FilterDataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from dataset import RetrievalDataset
import json
import fire
from tqdm import tqdm

def main(month: int = 9,
         data_root: str = 'QA_dataset',
         chunk: bool = False,
         chunk_size: str = 32,
         start_chunk_idx: int = 0,
         llama_config_dir: str = 'meta-llama/Llama-2-7b',
         llama_ckpt_path: str = None,
         pred_ckpt_path: str = 'classification/ckpt',
         batch_size: int = 8,
         save_root: str = 'results',
         db_faiss_dir: str = 'vectorstore/sentbert',
         model_name: str = None,
         hp = 2.
        ):
    
    if model_name is None:
        if 'sentbert' in db_faiss_dir:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        elif 'contriever' in db_faiss_dir:
            model_name =D
        if 'e5' in db_faiss_dir:
            model_name = 'intfloat/e5-large-v2'
    db_type = db_faiss_dir.split('/')[-1]
    
    # init model
    accelerator = Accelerator()
    
    dataset = RetrievalDataset(month, 
                            data_root,
                            chunk=chunk,
                            chunk_size=chunk_size,
                            start_chunk_idx=start_chunk_idx
                            )    
            
    # retrieve
    retrievals = retrieve(accelerator,
                          month,
                          dataset,
                          chunk=chunk,
                          chunk_size=chunk_size,
                          start_chunk_idx=start_chunk_idx,
                          return_results=True,
                          save=True,
                          db_faiss_dir=db_faiss_dir,
                          model_name=model_name
                          )
    
    # first retrieve
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
    
    # inference
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

    with open(f'{save_root}_{db_type}/{month:02d}/all_.json', 'w') as f:
        json.dump(all, f, indent = 2)
        
    # select label
    output = {}
    reretrieve = {}
    for qid, qitem in all:
        if qid not in output:
            output[qid] = qitem
        else:
            if output[qid]['prediction']['prob'] < qitem['prediction']['prob']:
                output[qid] = qitem 
    
    for qid, qitem in output.items():
        if qitem['prediction']['label'] > 0:
            reretrieve[qid] = qitem
            
    with open(f'{save_root}_{db_type}/{month:02d}/selected_.json', 'w') as f:
        json.dump(output, f, indent = 2)
    with open(f'{save_root}_{db_type}/{month:02d}/reretrieve_.json', 'w') as f:
        json.dump(reretrieve, f, indent = 2)

    ### verification

    ### re-retrieve
    # prepare new dataset
    with open(f'{save_root}_{db_type}/{month:02d}/reretrieve.json') as f:
        reretrieve = json.load(f)
    
    dataset = RetrievalDataset(month, 
                               dataset=reretrieve,
                               chunk=False,
                               mode='reretrieve'
                               )
    retrievals = retrieve(accelerator,
                          month,
                          dataset,
                          chunk=False,
                          return_results=True,
                          save=False,
                          mode='reretrieve'
                          )
    
    with open(f'{save_root}_{db_type}/{month:02d}/reretrieve_input_{hp}.json') as f:
        retrievals = list(json.load(f).items())
    print(f'{save_root}_{db_type}/{month:02d}/reretrieve_input_{hp}.json')
    print(len(retrievals))
        
    tokenizer, model = get_model(accelerator, 
                                 llm_config_dir, 
                                 llm_ckpt_path, 
                                 train_pred=False, 
                                 pred_ckpt_path=pred_ckpt_path
                                 )
    accelerator.wait_for_everyone()
    model.eval()
    
    dataset = FilterDataset(dataset=retrievals, mode='test', query='reretrieval')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    dataloader = accelerator.prepare(dataloader)
    
    # inference
    all = []
    for batch in tqdm(dataloader):
        qaids, prompts, metas = batch
        outputs = model.inference(prompts)
        for qaid, out, meta in zip(qaids, outputs, metas):
            meta['old_prediction'] = meta['prediction']
            meta['prediction'] = out
            all.append([qaid, meta])
    all = gather_object(all)
        
    # select label
    output = {}
    for qid, qitem in all:
        if qid not in output:
            output[qid] = qitem
        else:
            if output[qid]['prediction']['prob'] < qitem['prediction']['prob']:
                output[qid] = qitem 
    
    with open(f'{save_root}_{db_type}/{month:02d}/reretrieve_output_{hp}.json', 'w') as f:
        json.dump(output, f, indent = 2)
    
    with open('09/sim_answer.json') as f: 
        retrievals = json.load(f)
        retrievals = list(retrievals.items())
        
    tokenizer, model = get_model(accelerator, 
                                 llm_config_dir, 
                                 llm_ckpt_path, 
                                 train_pred=False, 
                                 pred_ckpt_path=pred_ckpt_path
                                 )
    accelerator.wait_for_everyone()
    model.eval()
    
    dataset = FilterDataset(dataset=retrievals, mode='test', query='retrieval_update')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    dataloader = accelerator.prepare(dataloader)
    
    # inference
    all = []
    for batch in tqdm(dataloader):
        qaids, prompts, metas = batch
        outputs = model.inference(prompts)
        for qaid, out, meta in zip(qaids, outputs, metas):
            meta['prediction_update'] = out
            all.append([qaid, meta])
    all = gather_object(all)
        
    # select label
    output = {}
    for i, (qid, qitem) in enumerate(all):
        if i % 3 == 0:
            select = sorted(all[i : i+3], key=lambda x: x[1]['prediction_update']['prob'], reverse=True)[0][1]
            label = select['prediction_update']['label']
            select['label'] = label
            output[qid] = select

    
    with open('09/sim_answer_outputs.json') as f:
        json.dump(output, f, indent=2)
    
    
if __name__ == "__main__":
    fire.Fire(main)