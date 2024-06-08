from dataset import RetrievalDataset
from torch.utils.data import DataLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from string import Template
import os
import fire
import json
import torch
import warnings

warnings.filterwarnings("ignore")

def load_vectorstore(
        month: int = 8,
        db_root: str = 'vectorstore/sentbeert',
        normalize_L2: bool = False,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ) -> FAISS:
    
    # if vectorstore exists
    db_faiss_dir = f'{db_root}/{month:02d}'
    if os.path.exists(f'{db_faiss_dir}/index.faiss'):
        embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                           model_kwargs={'device': 'cuda'},
                                           encode_kwargs={'batch_size': 2048,
                                                         'show_progress_bar': False,
                                                         'device': 'cuda'
                                                         }
                                           )
        db = FAISS.load_local(db_faiss_dir, embeddings=embeddings) #, normalize_L2=normalize_L2)
        return db
    else:
        raise Exception(f'DB directory {db_faiss_dir} is invalid.')


class Retrieval:
    
    def __init__(
        self,
        db,
        search_kwargs=None,
    ):
        self.is_ralm = True
        self.db = db
        self.search_kwargs = search_kwargs
        
    def _get_context(self, inputs):
        docs = self.db.similarity_search_with_score(inputs['query'], **self.search_kwargs)
        retrieved = []
        context = ''
        gold = inputs['grounded_sentence']
        for doc, score in docs:
            # record retrieval
            doc.metadata['document'] = doc.page_content
            doc.metadata['score'] = round(float(score), 2)
            # set hit
            hit = 0
            if gold in doc.page_content:
                hit = 1
            doc.metadata['hit'] = hit
            # save
            retrieved.append(doc.metadata)
            context = context + doc.page_content + '\n\n' 
        #retreived = sorted(retrieved, sor)
        inputs['context'] = context[:-2]
        inputs['retrieval'] = retrieved
        return inputs
    
    def __call__(self, inputs):
        return self._get_context(inputs)   
        
         
def retrieve(
        accelerator: Accelerator = None,
        month: int = 8,
        dataset: RetrievalDataset = None,
        dataset_dir: str = 'dialogues',
        save: bool = True,
        save_root: str = 'datasets',
        db_faiss_dir: str = '../inference_qa_ralm/vectorstore/sentbert',
        chunk: bool = False,
        chunk_size: int = 1600,
        start_chunk_idx: int = 0 ,
        return_results: bool = False,
        top_k: int = 3,
        model_name: str = None
    ):
    
    
    # make chain class
    retriever_db = load_vectorstore(month, db_faiss_dir, model_name=model_name)
    retrieval = Retrieval(retriever_db, search_kwargs={"k":top_k})
    
    if dataset is None and dataset_dir:
        dataset = RetrievalDataset(month, 
                                root_dir=dataset_dir,
                                chunk=chunk,
                                chunk_size=chunk_size,
                                start_chunk_idx=start_chunk_idx
                                ) 
        
    # get dataset
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    if accelerator is not None:
        dataloader = accelerator.prepare(dataloader)
    
    # path to save
    db_type = db_faiss_dir.split('/')[-1]
    if chunk:
        save_dir = f'{save_root}/{db_type}/{month:02d}/retrievalturn_{start_chunk_idx}.json'
    else:
        save_dir = f'{save_root}/{db_type}/{month:02d}/retrievalturn.json'
        
    # execute inference
    results = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        tid = batch[0]
        turn = batch[1]
        output = retrieval(turn) 
        results.append([tid, output])
        if save:
            if accelerator is not None:
                with open(save_dir.replace('.json', f'{torch.distributed.get_rank()}.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps({tid: output}) + '\n')
            else:
                with open(save_dir.replace('.json', '.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps({tid: output}) + '\n')
        
    # gather results
    if accelerator is not None:
        results = gather_object(results)
    results = sorted(results, key=lambda x: int(x[0]))
    
    # save results
    if save:
        results = dict(results)
        with open(save_dir, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved in {save_dir}')
        
    if return_results:
        return results


if __name__ == "__main__":
    fire.Fire(retrieve)