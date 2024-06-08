from retrievals.bm25 import BM25
from retrievals.simcse import SIMCSE    

__all__ = ['BM25', 'SIMCSE']

retrieval_classes = {
    "bm25": BM25,
    "simcse": SIMCSE,
}

def get_retrieval(retrieval: str):
    # check if retrieval method is supported
    if retrieval not in retrieval_classes:
        raise ValueError(f"Retrieval method {retrieval} not supported, please choose from {list(retrieval_classes.keys())}")
    
    return globals()[retrieval_classes[retrieval]]()