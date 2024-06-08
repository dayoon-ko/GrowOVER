from typing import List
from retrievals.base_retrieval import BaseRetrieval

class SIMCSE(BaseRetrieval):
    def __init__(self):
        raise NotImplementedError 
    
    def retrieve(self, query: str):
        raise NotImplementedError
    
    def retrieve_batch(self, query: List[str]):
        raise NotImplementedError