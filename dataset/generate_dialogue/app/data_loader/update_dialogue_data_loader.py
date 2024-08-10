from argparse import Namespace
from typing import Tuple, Dict

from app.data_loader import BaseDataLoader


class UpdateDialogueDataLoader(BaseDataLoader):
    def __init__(self, args: Namespace):
        self.wiki_dir = args.wiki_dir
        self.label_dir = args.label_dir
        self.dialogue_dir = args.dialogue_dir
        self.month = args.month
        self.old_month = args.month - 1
        
        self.wiki_data, self.article_ids = self._get_wiki_data(self.wiki_dir, self.month)
        self.old_label_data = self._get_label_data(self.label_dir, self.old_month)
        self.new_label_data = self._get_label_data(self.label_dir, self.month)
        
        self.old_article_dialogue_data = self._get_article_dialogue_data(self.dialogue_dir, self.old_month)
        self.old_dialogue_data = self._get_dialogue_data(self.dialogue_dir, self.old_month)
        self.old_turn_data = self._get_turn_data(self.dialogue_dir, self.old_month)
        
        
        self.dialogue_id = max(map(int, self.old_dialogue_data.keys())) + 1 if self.old_dialogue_data else 0
        self.turn_id = max(map(int, self.old_turn_data.keys())) + 1 if self.old_turn_data else 0
        
        self.now = 0
        self.end = len(self.article_ids)
    
    def __next__(self) -> Tuple[str, Dict, Dict, Dict, Dict]:
        if self.now >= self.end:
            raise StopIteration()

        article_id = self.article_ids[self.now]
        wiki = self.wiki_data[article_id]
        old_label = self._get_label(article_id, self.old_label_data)
        new_label = self._get_label(article_id, self.new_label_data)
        old_dialogue, old_turn = self._get_dialogue(article_id)
        self.now += 1
        return article_id, wiki, old_label, new_label, old_dialogue, old_turn