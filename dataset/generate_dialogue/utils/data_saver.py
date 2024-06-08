import os
import json
from typing import Dict

class DataSaver:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
    
    
    def save(self, data: Dict, data_path: str, data_root: str):
        #--- get save path ---#
        save_path = self.get_save_path(data_path, data_root)
        
        #--- save ---#
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    
    def get_save_path(self, data_path: str, data_root: str) -> str:
        # get save path from data path e.g., ['AA', 'wiki_00.json']
        folder, file = data_path[len(data_root)+1:].split("/")
        
        # make save directory if not exists
        save_dir = os.path.join(self.save_dir, folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # make save path
        save_path = os.path.join(save_dir, file)
        
        return save_path