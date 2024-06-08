import json
from typing import Dict, Tuple
from pathlib import Path


class DataLoader:
    def __init__(self, data_root: str):
        dir = Path(data_root)
        self.files = [str(i) for i in sorted(dir.glob('*/wiki_*.json'))]
        self.curr_idx = 0
        self.max_idx = len(self.files)
    
    def __iter__(self) -> Tuple[Dict, str]:
        return self
    
    def __next__(self) -> Tuple[Dict, str]:
        #--- check if all files are read ---#
        if self.curr_idx >= self.max_idx:
            raise StopIteration
        
        #--- read json ---#
        file = self.files[self.curr_idx]
        with open(file) as f:
            js = json.load(f)
            
        #--- update index ---#
        self.curr_idx += 1
        
        return js, file