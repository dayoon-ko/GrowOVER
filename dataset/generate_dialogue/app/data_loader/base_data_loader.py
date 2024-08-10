import os
import jsonlines
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, Tuple, List


class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, args: Namespace):
        ...
    
    @abstractmethod
    def __next__(self):
        ...
    
    def __iter__(self):
        return self
    
    def _get_wiki_data(self, dir: str, month: int) -> Tuple[Dict, List]:
        file_path = self._get_wiki_path(dir, month)
        wiki_data = self._get_data(file_path)
        article_id = list(map(str, sorted(map(int, wiki_data.keys()))))
                
        return wiki_data, article_id
    
    def _get_label_data(self, dir: str, month: int) -> Dict:
        file_path = self._get_label_path(dir, month)
        return self._get_data(file_path)
    
    def _get_dialogue_data(self, dir: str, month: int) -> Dict:
        file_path = self._get_dialogue_path(dir, month)
        return self._get_data(file_path)
    
    def _get_turn_data(self, dir: str, month: int) -> Dict:
        file_path = self._get_turn_path(dir, month)
        return self._get_data(file_path)
    
    def _get_article_dialogue_data(self, dir: str, month: int) -> Dict:
        file_path = self._get_article_dialogue_path(dir, month)
        return self._get_data(file_path)
    
    def _get_wiki_path(self, dir: str, month: int) -> str:
        return os.path.join(dir, f"{month:02}/text.jsonl")
    
    def _get_label_path(self, dir: str, month: int) -> str:
        return os.path.join(dir, f"{self.old_month:02}{self.month:02}/{month:02}/result.jsonl")
    
    def _get_dialogue_path(self, dir: str, month: int) -> str:
        return os.path.join(dir, f"{month:02}/dialogue.jsonl")
    
    def _get_turn_path(self, dir: str, month: int) -> str:
        return os.path.join(dir, f"{month:02}/turn.jsonl")
    
    def _get_article_dialogue_path(self, dir: str, month: int) -> str:
        return os.path.join(dir, f"{month:02}/article_dialogue.jsonl")
    
    def _get_data(self, file: str) -> Dict:
        reader = jsonlines.open(file)
        result = dict()
        for data in reader:
            id = list(data.keys())[0]
            result[id] = data[id]
        reader.close()
        return result
    
    def _get_label(self, article_id: str, label_data: Dict) -> Dict:
        if article_id in label_data:
            return label_data[article_id]
        return None
    
    def _get_dialogue(self, article_id: str) -> Dict:
        dialogue, turn = {}, {}
        if article_id not in self.old_article_dialogue_data:
            return dialogue, turn
        
        dialogue_ids = self.old_article_dialogue_data[article_id]
        for dialogue_id in dialogue_ids:
            dialogue[dialogue_id] = self.old_dialogue_data[dialogue_id]
            for _, turn_id in dialogue[dialogue_id]["turn"].items():
                if turn_id not in self.old_turn_data:
                    continue
                turn[turn_id] = self.old_turn_data[turn_id]
        return dialogue, turn