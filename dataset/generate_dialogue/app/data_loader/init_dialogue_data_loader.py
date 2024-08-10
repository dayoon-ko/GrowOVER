from argparse import Namespace
from typing import Tuple, Dict

from app.data_loader import BaseDataLoader


class InitDialogueDataLoader(BaseDataLoader):
    def __init__(self, args: Namespace):
        super().__init__(args)
        
        self.wiki_dir = args.wiki_dir
        self.dialogue_dir = args.dialogue_dir
        self.month = args.month
        
        self.wiki_data, self.article_ids = self._get_wiki_data(self.wiki_dir, self.month)
        
        self.dialogue_id, self.turn_id = 0, 0
        
        self.now = 0
        self.end = len(self.article_ids)
       
    
    def __next__(self) -> Tuple[str, Dict]:
        if self.now >= self.end:
            raise StopIteration()
        
        id = self.article_ids[self.now]
        wiki = self.wiki_data[id]
        self.now += 1
        return id, wiki