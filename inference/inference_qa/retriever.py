from dataset import RetrievalDataset
from torch.utils.data import DataLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from accelerate import Accelerator
from accelerate.utils import gather_object
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from string import Template
import os
import fire
import json
import warnings

warnings.filterwarnings("ignore")

def load_vectorstore(
        month: int = 8,
        db_root: str = 'vectorstore',
        normalize_L2: bool = False,
        model_name: str = None
    ) -> FAISS:
    
    # if vectorstore exists
    db_faiss_dir = f'{db_root}/{month:02d}'
    if os.path.exists(f'{db_faiss_dir}/index.faiss'):
        embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                           model_kwargs={'device': 'cuda'},
                                           encode_kwargs={'batch_size': 2048,
                                                         'device': 'cuda'
                                                         }
                                           )
        db = FAISS.load_local(db_faiss_dir, embeddings=embeddings, allow_dangerous_deserialization=True) #, normalize_L2=normalize_L2)
        print(db_faiss_dir)
        return db
    else:
        raise Exception(f'DB directory {db_faiss_dir} is invalid.')


class Retrieval:
    
    def __init__(
        self,
        db,
        mode="retrieve",
        query_key="question",
        search_kwargs=None,
    ):
        self.db = db
        self.mode = mode
        self.query_key = query_key
        self.search_kwargs = search_kwargs
        
    def _get_context(self, inputs):
        docs = self.db.similarity_search_with_score(inputs[self.query_key], **self.search_kwargs)
        retrieved = []
        context = ''
        gold = inputs['grounded_text']
        for doc, score in docs:
            # record retrieval
            doc.metadata['document'] = doc.page_content
            doc.metadata['score'] = round(float(score), 2)
            # set hit
            hit = 1
            for i in gold:
                if i not in doc.page_content:
                    hit = 0
            doc.metadata['hit'] = hit
            # save
            retrieved.append(doc.metadata)
            context = context + doc.page_content + '\n\n' 
        inputs['context'] = context[:-2]
        if self.mode == "retrieve":
            inputs['retrieval'] = retrieved
        if self.mode == "reretrieve":
            inputs['reretrieval'] = retrieved
        return inputs
    
    def __call__(self, inputs):
        return self._get_context(inputs)   
        


def retrieve(
        accelerator: Accelerator,
        month: int = 8,
        dataset: RetrievalDataset = None,
        save: bool = True,
        save_root: str = 'QA_dataset',
        db_faiss_dir: str = 'vectorstore',
        return_results: bool = False,
        top_k: int = 3,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
    
        
    # make chain class
    retriever_db = load_vectorstore(month, db_faiss_dir, model_name=model_name)
    retrieval = Retrieval(retriever_db, mode="retrieve", search_kwargs={"k":top_k})
    
    # get dataset
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    dataloader = accelerator.prepare(dataloader)
    
    # save path
    if save:
        save_pth = f"{save_root}/{month:02d}/retrieval.jsonl"
        
    # execute inference
    results = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        qid = batch[0]
        qa = batch[1]
        output = retrieval(qa) 
        output['id'] = qid
        results.append([qid, output])
        
    # gather results
    results = gather_object(results)
    results = sorted(results, key=lambda x: int(x[0]))
    
    # save results
    if save:
        with open(save_pth, 'w') as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        print(f'Saved in {save_pth}')
        
    if return_results:
        return results
    
        
def reretrieve(
        accelerator: Accelerator,
        month: int = 8,
        dataset: RetrievalDataset = None,
        save: bool = True,
        save_root: str = 'QA_dataset',
        db_faiss_dir: str = 'vectorstore',
        return_results: bool = False,
        top_k: int = 3,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        num_candidates: int = 20,
        hp_lambda: float = 2.
    ):
    
    # model
    model = SentenceTransformer(model_name)
    
    # make chain class
    retriever_db = load_vectorstore(month, db_faiss_dir, model_name=model_name)
    retrieval = Retrieval(retriever_db, mode="reretrieve", search_kwargs={"k":top_k + num_candidates})
    
    # get dataset
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=dataset.collate_fn
                            )
    dataloader = accelerator.prepare(dataloader)
    
    # execute inference
    results = []
    for qid, qitem in tqdm(dataloader, total=len(dataloader)):
        qs = qitem['question']
        qas = qitem['question'] + ' ' + qitem['prediction']['text']
        omega = qitem['prediction']['prob'] * hp_lambda
        reretrieval = retrieval(qitem)["reretrieval"][3:]
        
        # Encode queries and documents
        queries = [qs, qas]
        docs = [r['document'] for r in reretrieval]
        if 'e5' in model_name:
            queries = ['query: ' + i for i in queries]
            docs = ['passage: ' + i for  i in docs]
        emb_q = model.encode(queries, convert_to_tensor=True)
        emb_d = model.encode(docs, convert_to_tensor=True)
        
        # Get cosine similarity
        if 'e5' in model_name:
            sim =(emb_q @ emb_d.transpose(1,0)).tolist()
        else:
            sim = util.pytorch_cos_sim(emb_q, emb_d).tolist()
        
        # Sort with cosine similarity
        output_ = []
        for r, sim_q, sim_qa in zip(reretrieval, sim[0], sim[1]):
            output_.append({'output': r,
                            'sim': (1 - omega) * sim_q + omega * sim_qa})
        output_ = sorted(output_, key=lambda x: x['sim'], reverse=True)[:3]
        qitem['reretrieval'] = output_
        
        results.append([qid, qitem])
        
    # Get the final results  
    if return_results:
        return results


if __name__ == "__main__":
    fire.Fire(retrieve)
    
    