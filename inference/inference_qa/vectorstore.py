from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import fire
import torch
from tqdm import tqdm
from glob import glob


def store_data(
        month: int = 8,
        glob_dir:str = '*', #'A*/wiki*',
        data_dir: str = 'snapshots_experiments',
        db_faiss_dir: str = 'vectorstore',
        batch_size: int = 1024,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
    
    db_faiss_dir = f"{db_faiss_dir}/{month:02d}"
    data_dir = f"{data_dir}/{month:02d}"
    
    # Document
    loader = DirectoryLoader(data_dir, glob=glob_dir, loader_cls=JSONLoader, loader_kwargs={'jq_schema':'.text', 'json_lines':True})
    documents = loader.load()
    print(f'Document count: {len(documents)}')
    
    # Split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=10)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                        #multi_process=False,
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'batch_size': batch_size,
                                                        'device': 'cuda'
                                                        }
                                        )    
    # Make a DB
    print(f'Extract db from documents {db_faiss_dir}')
    db = FAISS.from_documents(splits, embeddings)
    print(f'Saving embeddings to {db_faiss_dir}')
    db.save_local(f'{db_faiss_dir}')
    print('Saved')
        
        
def load_vectorstore(
        month: int = 8,
        db_root: str = 'vectorstore',
        normalize_L2: bool = False
    ) -> FAISS:
    
    # if vectorstore exists
    db_faiss_dir = f'{db_root}/{month:02d}'
    if os.path.exists(f'{db_faiss_dir}/index.faiss'):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                           model_kwargs={'device': 'cuda'},
                                           encode_kwargs={'batch_size': 2048,
                                                         'device': 'cuda'
                                                         }
                                           )
        db = FAISS.load_local(db_faiss_dir, embeddings=embeddings) #, normalize_L2=normalize_L2)
        return db
    # elif partitioned, merge  
    elif os.path.exits(db_faiss_dir):
        db = merge_vectorstore(month, db_root, False)
        return db
    else:
        raise Exception(f'DB directory {db_faiss_dir} is invalid.')


def merge_vectorstore(
        month: int = 8,
        db_root: str = 'vectorstore',
        save: bool = True
    )-> FAISS:
    db_dir = f'{db_root}/{month:02d}/*'
    dirs = [i for i in sorted(glob(db_dir)) if os.path.isdir(i)]
    # load first db
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                           model_kwargs={'device': 'cuda'},
                                           encode_kwargs={'batch_size': 2048,
                                                         'device': 'cuda'
                                                         }
                                         )
    total_db = FAISS.load_local(dirs[0], embeddings=embeddings)
    for dir in tqdm(dirs[1:]):
        db = FAISS.load_local(dir, embeddings)
        total_db.merge_from(db)
    
    if save:
        total_db.save_local(f'{db_root}/{month:02d}/db')
        print('Success!')
    else:
        return total_db 
    
    
if __name__ == "__main__":
    fire.Fire(store_data)