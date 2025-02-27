from torch.utils.data import Dataset
import json
from typing import Optional, Union
from glob import glob

class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        month: int = 8,
        root_dir: str = 'QA_dataset',
        dataset: Union[list, dict] = None,
        mode: str = 'retrieve'
    ):  
        self.mode = mode
        if dataset:
            if type(dataset) == dict:
                dataset = list(dataset.items())
            self.data = dataset
        else:
            self.month = month
            self.data_path = f'{root_dir}/{month:02d}/qa.json'
            with open(self.data_path) as f:
                js = json.load(f)
            print(f'Total {len(js)} data points')    
            self.data = sorted(list(js.items()), key=lambda x: int(x[0]))
        
    def __getitem__(self, idx):
        datapoint = self.data[idx]
        qaid = datapoint[0]
        meta = datapoint[1]
        return qaid, meta
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        return batch[0]

