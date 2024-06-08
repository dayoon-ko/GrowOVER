from abc import ABC, abstractmethod
from typing import List

class BaseRetrieval(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def retrieve(self, query: str):
        pass
    
    @abstractmethod
    def retrieve_batch(self, query: List[str]):
        pass