from torch.utils.data import Dataset
import json
from typing import Optional, Union
from glob import glob

class RetrievalDataset(Dataset):
    
    def __init__(
        self,
        month: int = 8,
        root_dir: str = '/gallery_moma/hahyeon.choi/TemporalBenchmark/dialogues',
        dataset: Union[list, dict] = None,
    ):  
        if dataset:
            if type(dataset) == dict:
                dataset = list(dataset.items())
            self.data = dataset
        else:
            self.dialogue_path = f'{root_dir}/{month:02d}/dialogue.jsonl'
            self.turn_path = f'{root_dir}/{month:02d}/turn.jsonl'
            self.dialogues = self._get_json(self.dialogue_path)
            self.dialogues = sorted(list(self.dialogues.items()), key=lambda x: x[0])
            self.turns = self._get_json(self.turn_path)
            self.data = self._get_data()[:8]
            
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        return batch[0]
    
    def _get_data(self):
        data = []
        for idx in range(len(self.dialogues)):
            dial_id, dialogue_meta = self.dialogues[idx]
            article_id = dialogue_meta['article_id']
            query = ''
            history = ''
            for _, turn_id in sorted(dialogue_meta['turn'].items(), key=lambda x: int(x[0])):
                try:
                    turn_meta = self.turns[turn_id]
                except:
                    continue
                query += ' ' + turn_meta['user'] 
                turn_meta['query'] = query
                turn_meta['history'] = history
                data.append([
                    turn_id,
                    turn_meta,
                ])
                query += ' ' + turn_meta['expert'] 
                history += 'User: ' + turn_meta['user'] + '\n' + 'Expert: ' + turn_meta['expert'] + '\n'

        return data
    
    def _get_json(self, jsonl_fn):
        js = {}
        with open(jsonl_fn) as f:
            js_list = [json.loads(i) for i in f.readlines()]
            for i in js_list:
                js.update(i)
        return js
