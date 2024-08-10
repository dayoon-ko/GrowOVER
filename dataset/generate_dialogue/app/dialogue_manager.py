from typing import Dict, Tuple, Set, List
from argparse import Namespace

from app.utils import dialogue_types


class DialogueManager:
    def __init__(self, args: Namespace):
        self.month = args.month
        
    
    def update_dialogue(self, title: str, old_dialogue: Dict, old_turn: Dict, old_label: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
            check it reference sentences are maintained or deleted
            if maintained, update sentence index
            elif deleted, update dialogue as deleted
        """
        maintained_dialogue, maintained_turn = {}, {}
        deleted_dialogue, deleted_turn = {}, {}
        for dialogue_id, dialogue in old_dialogue.items():
            count = len(dialogue["turn"])
            for _, turn_id in dialogue["turn"].items():
                if turn_id not in old_turn:
                    # This turn has already been deleted
                    count -= 1
                    continue
        
                turn = old_turn[turn_id]
                sentence_index = turn["sentence_index"]
                if sentence_index in old_label["S"]:
                    turn = self._update_turn_as_same(turn, str(old_label["S"][sentence_index]))
                    maintained_turn[turn_id] = turn
                else: # Deleted & Unlabelled Sentences are deleted
                    count -= 1
                    turn = self._update_turn_as_delete(turn)
                    deleted_turn[turn_id] = turn
        
            if count == 0:
                dialogue = self._update_dialogue_as_delete(dialogue)
                deleted_dialogue[dialogue_id] = dialogue
            else:
                dialogue = self._update_dialogue_as_same(dialogue)
                maintained_dialogue[dialogue_id] = dialogue
                
        return maintained_dialogue, maintained_turn, deleted_dialogue, deleted_turn
    
    
    def _update_turn_as_same(self, turn: Dict, new_sentence_index: str) -> Dict:
        turn['sentence_index'] = new_sentence_index
        turn['sentence_type'] = dialogue_types["same"]
        return turn
    
    
    def _update_turn_as_delete(self, turn: Dict) -> Dict:
        turn["deleted_date"] = self.month
        turn["sentence_index"] = str(-1)
        turn["sentence_type"] = dialogue_types["deleted"]
        return turn
    
    def _update_dialogue_as_same(self, dialogue: Dict) -> Dict:
        dialogue["dialogue_type"] = dialogue_types["same"]
        dialogue["last_modified_month"] = self.month
        return dialogue
    
    def _update_dialogue_as_delete(self, dialogue: Dict) -> Dict:
        dialogue["dialogue_type"] = dialogue_types["deleted"]
        dialogue["last_modified_month"] = self.month
        return dialogue