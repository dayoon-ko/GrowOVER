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
import os 

def main(month: int = 9,
         data_root: str = 'datasets',
         chunk: bool = False,
         chunk_size: str = 4,
         start_chunk_idx: int = 0,
         llm_config_dir: str = None, 
         llm_ckpt_path: str = None,
         pred_ckpt_path: str = None,
         batch_size: int = 8,
         save_root: str = 'results',
         db_faiss_dir: str = '../inference_qa_ralm/vectorstore/sentbert',
         model_name: str = None,
         save: bool = False,
         retrieval_path: str = None,
         concat: bool = False,
         pred_new_ckpt_path: str = None, 
         reverse: bool = False,
        ):
    
    # init model
    accelerator = Accelerator()
    
    if model_name is None:
        if 'sentbert' in db_faiss_dir:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        elif 'contriever' in db_faiss_dir:
            model_name = 'facebook/contriever'
        if 'e5' in db_faiss_dir:
            model_name = 'intfloat/e5-large-v2'
    
    if pred_new_ckpt_path is not None:
        pred_ckpt_path = pred_new_ckpt_path
        
    db_type = db_faiss_dir.split('/')[-1]
    
    if retrieval_path is None:
        retrieval_path = f'datasets/{db_type}/{month:02d}/retrievalturn.jsonl'
        if os.path.exists(retrieval_path):
            with open(retrieval_path) as f: 
                retrievals = [json.loads(i) for i in f.readlines()]
                retrievals = [[k, v] for ret in retrievals for k, v in ret.items()]
                retrievals = sorted(retrievals, key=lambda x: int(x[0]), reverse=reverse)
                
        else:
            dataset = RetrievalDataset(month, 
                                    #data_root,
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
                                save=save,
                                model_name=model_name
                                )
            print(len(retrievals), 'datapoints with retrieved documents')

            if save:
                with open(retrieval_path) as f:
                    json.dump(retrievals, f, indent=2)
            
    else:
        with open(retrieval_path) as f:
            retrievals = [json.loads(i) for i in f.readlines()]
            retrievals = [[k, v] for ret in retrievals for k, v in ret.items()]
            retrievals = sorted(retrievals, key=lambda x: int(x[0]))
    
    db_type = db_faiss_dir.split('/')[-1]
    if concat:
        db_type = f'{db_type}_concat'
        print(db_type)
    if pred_new_ckpt_path is not None:
        db_type = f'{db_type}_retry'
        
    # if already done, remove
    save_interval_path = f'results_{db_type}/{month:02d}/all.jsonl'
    if '08' not in llm_ckpt_path:
        save_interval_path = f'results_{db_type}/{month:02d}/concat_cp.jsonl'
    elif concat:
        save_interval_path = f'results_{db_type}/{month:02d}/concat.jsonl'
    print(save_interval_path)
    if os.path.exists(save_interval_path):
        with open(save_interval_path, errors="ignore") as f:
            #js = [json.loads(i) for i in f.readlines()]
            lines = f.readlines()
            print(len(lines))
            js = []
            for i, l in enumerate(lines):
                try:
                    l = json.loads(l)
                    if type(l) == dict:
                        js.append(l)
                    else:
                        print(i, l)
                        exit()
                except:
                    continue
            done = sorted([k for i in js for k, v in i.items()],
                          key = lambda x: int(x[0]))
            print(len(done), 'are done.')
            done_id = []
            if not concat:
                for i, qid in enumerate(done[:-2]):
                    if done[i+1] == qid and done[i+2] == qid:
                        done_id.append(qid)
            #else:
            done_id = set(done)
            todo_id = set([i[0] for i in retrievals])
            left_id = todo_id - done_id
            
        retrievals = [i for i in retrievals if i[0] in left_id]
        print(retrievals[0][0])

    '''
    save_interval_path = f'results_{db_type}/{month:02d}/all.jsonl'
    if os.path.exists(save_interval_path):
        with open(save_interval_path) as f:
            done_id = sorted([k for i in f.readlines() for k, v in json.loads(i).items()],
                             key=lambda x: int(x))
        retrievals_left = []
        for k, v in retrievals:
            if k not in done_id:
                retrievals_left.append([k, v])
        retrievals = retrievals_left
        del retrievals_left
           
    print(len(retrievals))
    '''
    # first retrieve
    tokenizer, model = get_model(accelerator, 
                                 llm_config_dir, 
                                 llm_ckpt_path, 
                                 train_pred=False, 
                                 pred_ckpt_path=pred_ckpt_path
                                 )
    accelerator.wait_for_everyone()
    model.eval()
    
    dataset = FilterDataset(dataset=retrievals, mode='test', concat=concat)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    dataloader = accelerator.prepare(dataloader)
    
    # inference
    all = []
    for batch in tqdm(dataloader):
        qaids, questions, metas = batch
        outputs = model.inference(questions)
        for qaid, out, meta in zip(qaids, outputs, metas):
            meta['prediction'] = out
            all.append([qaid, meta])
            with open(save_interval_path, 'a') as f:
                f.write(json.dumps({qaid:meta}) + '\n')
            
    all = gather_object(all)
        
    with open(f'results_{db_type}/{month:02d}/all.json', 'w') as f:
        json.dump(all, f, indent = 2)
        
    # select label
    output = {}
    reretrieve = {}
    for i, (qid, qitem) in enumerate(all):
        if i % 3 == 0:
            select = sorted(all[i : i+3], key=lambda x: x[1]['prediction']['prob'], reverse=True)[0][1]
            output[qid] = select
            if select['prediction']['label'] != 0:
                reretrieve[qid] = select

    with open(f'results_{db_type}/{month:02d}/selected.json', 'w') as f:
        json.dump(output, f, indent = 2)
    with open(f'results_{db_type}/{month:02d}/reretrieve.json', 'w') as f:
        json.dump(reretrieve, f, indent = 2)
        
    ### re-retrieve
    '''
    # prepare new dataset
    dataset = RetrievalDataset(month, 
                               dataset=reretrieve,
                               chunk=False
                               )
    
    retrievals = retrieve(accelerator,
                          month,
                          dataset,
                          chunk=False,
                          return_results=True,
                          save=False
                          )
    
    # re-retrieve
    '''
    
if __name__ == "__main__":
    fire.Fire(main)