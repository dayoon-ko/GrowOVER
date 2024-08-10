import os
import jsonlines
from typing import Dict, List
from argparse import Namespace

class DataSaver:
    def __init__(self, args: Namespace, deleted: bool = False):
        self.deleted = deleted
        self.save_dir = args.dialogue_dir
        self.month = args.month

        self._set_save_dir()
        
        self.article_dialogue_path = os.path.join(self.save_folder, f"article_dialogue.jsonl")
        self.dialogue_path = os.path.join(self.save_folder, f"dialogue.jsonl")
        self.turn_path = os.path.join(self.save_folder, f"turn.jsonl")
        
        self.article_dialogue_writer = jsonlines.open(self.article_dialogue_path, "w", flush=True)
        self.dialogue_writer = jsonlines.open(self.dialogue_path, "w", flush=True)
        self.turn_writer = jsonlines.open(self.turn_path, "w", flush=True)
        
        self.article_dialogue = dict()
        
    
    def __del__(self):
        self._save_article_dialogue()
        
        self.dialogue_writer.close()
        self.turn_writer.close()
        
    
    def _set_save_dir(self):
        self.save_folder = os.path.join(self.save_dir, f"{self.month:02}")
        if self.deleted:
            self.save_folder = os.path.join(self.save_folder, "deleted")
    
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    
    
    def set_id(self, dialogue_id: int, turn_id: int):
        self.dialogue_id = dialogue_id
        self.turn_id = turn_id
    
    
    def save_updated_dialogue(self, dialogues: Dict, turns: Dict):
        for dialogue_id, dialogue in dialogues.items():
            self.dialogue_writer.write({dialogue_id: dialogue})
            self._set_article_dialogue(dialogue["article_id"], dialogue_id)
                
        for turn_id, turn in turns.items():
            self.turn_writer.write({turn_id: turn})
    
    
    def save_created_dialogue(self, dialogues: List):
        for dialogue in dialogues:
            turn_match = {}
            for num, turn in enumerate(dialogue["dialogue"]):
                _turn = self._set_turn(num, turn)
                self.turn_writer.write({str(self.turn_id): _turn})
                turn_match[num] = str(self.turn_id)
                self.turn_id += 1
            
            _dialogue = self._set_dialogue(dialogue, turn_match)
            self._set_article_dialogue(_dialogue["article_id"], str(self.dialogue_id))
            self.dialogue_writer.write({str(self.dialogue_id): _dialogue})
            self.dialogue_id += 1
            
            
    def _set_turn(self, num: int, turn: Dict) -> Dict:
        # user, expert, sentence_type, sentence_index, grounded_sentence
        return {
            "dialogue_id": str(self.dialogue_id),
            "turn_number": str(num),
            "user": turn["user"],
            "expert": turn["expert"],
            "sentence_type": turn["sentence_type"],
            "sentence_index": turn["sentence_index"],
            "grounded_sentence": turn["grounded_sentence"],
        }

    
    def _set_dialogue(self, dialogue: Dict, turn_match: Dict) -> Dict:
        # article_id, created_month, last_modified_month, dialogue_type, turn
        return {
            "article_id": dialogue["article_id"],
            "created_month": dialogue["created_month"],
            "last_modified_month": dialogue["last_modified_month"],
            "dialogue_type": dialogue["dialogue_type"],
            "turn": turn_match,
        }
        
    
    def _set_article_dialogue(self, article_id: str, dialogue_id: str):
        if article_id not in self.article_dialogue:
            self.article_dialogue[article_id] = set()
        self.article_dialogue[article_id].add(dialogue_id)
        
    
    def _save_article_dialogue(self):
        for article_id, dialogue_ids in self.article_dialogue.items():
            self.article_dialogue_writer.write({article_id: list(dialogue_ids)})