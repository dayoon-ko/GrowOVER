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
         llm_config_dir: str = '/gallery_louvre/dayoon.ko/research/llama2/checkpoints',
         llm_ckpt_path: str = '/gallery_louvre/dayoon.ko/research/kotoba-recipes/checkpoints_08/init/iter_0000003/model.pt',
         pred_ckpt_path: str = '/gallery_louvre/dayoon.ko/research/kotoba-recipes/inference_dial_ralm/classification/checkpoints/lr_0.0001_wd_1e-07_20_0/9',
         #'/gallery_louvre/dayoon.ko/research/kotoba-recipes/inference_dial_ralm/classification/ckpts/lr_1e-05_wd_1e-07_5_0.3/19',
         #'/gallery_louvre/dayoon.ko/research/kotoba-recipes/inference_dial_ralm/classification/ckpts/lr_0.0001_wd_1e-07_5_0.3/5',
         batch_size: int = 8,
         save_root: str = 'results',
         db_faiss_dir: str = '../inference_qa_ralm/vectorstore/sentbert',
         model_name: str = None,
         save: bool = False,
         retrieval_path: str = None,
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
    
    db_type = db_faiss_dir.split('/')[-1]
    
    if retrieval_path is None:
        retrieval_path = f'results_{db_type}/{month:02d}/reretrieve_input.json'
    
    with open(retrieval_path) as f: 
        retrievals = json.load(f)
        retrievals = [[k, v] for k, v in retrievals.items()]
        retrievals = sorted(retrievals, key=lambda x: int(x[0]))

    db_type = db_faiss_dir.split('/')[-1]
        
    # if already done, remove
    save_interval_path = f'results_{db_type}/{month:02d}/reretrieve_output.jsonl'
    if os.path.exists(save_interval_path):
        with open(save_interval_path) as f:
            js = [json.loads(i) for i in f.readlines()]
            done = sorted([k for i in js for k, v in i.items()],
                          key = lambda x: int(x[0]))
            done_id = []
            for i, qid in enumerate(done[:-2]):
                if done[i+1] == qid and done[i+2] == qid:
                    done_id.append(qid)
            done_id = set(done_id)
            todo_id = set([i[0] for i in retrievals])
            left_id = todo_id - done_id
        retrievals = [i for i in retrievals if i[0] in left_id]
    
    # first retrieve
    tokenizer, model = get_model(accelerator, 
                                 llm_config_dir, 
                                 llm_ckpt_path, 
                                 train_pred=False, 
                                 pred_ckpt_path=pred_ckpt_path
                                 )
    accelerator.wait_for_everyone()
    model.eval()
    
    dataset = FilterDataset(dataset=retrievals, mode='test', concat=False, query='reretrieval')
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
        
    # select label
    output = {}
    for qid, qitem in all:
        if qid not in output:
            output[qid] = qitem
        else:
            if output[qid]['prediction']['prob'] < qitem['prediction']['prob']:
                output[qid] = qitem 
    with open(f'results_{db_type}/{month:02d}/reretrieve_output.json', 'w') as f:
        json.dump(output, f, indent = 2)
        
        
if __name__ == "__main__":
    fire.Fire(main)