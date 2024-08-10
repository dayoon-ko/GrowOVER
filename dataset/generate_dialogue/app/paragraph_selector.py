import random
import logging
from argparse import Namespace
from typing import List, Set, Dict, Tuple

from app.utils import split_article_into_sentences, dialogue_types

class ParagraphSelector:
    def __init__(self, args: Namespace):
        self.num_of_dialogue = args.num_of_dialogue
        self.num_of_new_dialogue = args.num_of_new_dialogue
        self.num_of_contradict_dialogue = args.num_of_contradict_dialogue
        self.min_threshold = args.min_threshold
        self.max_threshold = args.max_threshold
        
        self.same_label_idx, self.new_label_idx, self.contradict_label_idx = set(), set(), set()
        self.labelled_idx = set()
        
    
    def select_paragraphs_for_initialize(self, id: str, article: str) -> List[Tuple[List[str], int, List[str]]]:
        sentences, paragraph_indices = split_article_into_sentences(article)
        
        # sentences(sentence 0, sentence 1, ...), start_index (idx), type(type of sentence 0, type of sentehce 1, ...)
        candidate_paragraphs = []
        random.shuffle(paragraph_indices)
        
        for start_idx, end_idx in paragraph_indices:
            if len(candidate_paragraphs) == self.num_of_dialogue:
                break
            
            target_sentences, target_start_idx, target_end_idx = self._except_semi_colon_sentence(sentences[start_idx:end_idx], start_idx, end_idx)
            if self._is_not_informative_paragraph(target_sentences):
                continue
            sentences_type = [dialogue_types["new"] for _ in range(len(target_sentences))]
            candidate_paragraphs.append((target_sentences, target_start_idx, sentences_type))
        return candidate_paragraphs
            
            
    def select_paragraphs_for_update(self, article: str, new_label: Dict, old_label: Dict) -> List[Tuple[List[str], int, List[str]]]:
        sentences, paragraph_indices = split_article_into_sentences(article)
        
        # sentences(sentence 0, sentence 1, ...), start_index (idx), type(type of sentence 0, type of sentehce 1, ...)
        contradict_paragraphs, new_paragraphs = [], []
        
        self._set_sentences_label(new_label, old_label)
        random.shuffle(paragraph_indices)
        
        for start_idx, end_idx in paragraph_indices:
            if len(contradict_paragraphs) >= self.num_of_contradict_dialogue and len(new_paragraphs) >= self.num_of_new_dialogue:
                break
            
            target_sentences, target_start_idx, target_end_idx = self._except_semi_colon_sentence(sentences[start_idx:end_idx], start_idx, end_idx)
            sentences_type = self._get_sentences_type(target_start_idx, target_end_idx) # ["NEW", "CONTRADICT", "SAME"]
            
            if self._is_not_informative_paragraph(target_sentences):
                continue
            if self._is_not_updated_paragraph(sentences_type):
                continue
            if self._is_unlabelled_paragraph(target_start_idx, target_end_idx):
                continue
            
            if dialogue_types["contradict"] in sentences_type:
                contradict_paragraphs.append((target_sentences, target_start_idx, sentences_type))
            else:
                new_paragraphs.append((target_sentences, target_start_idx, sentences_type))
        
        candidate_paragraphs = self._final_candidate_paragraphs(contradict_paragraphs, new_paragraphs)
        return candidate_paragraphs
              
    
    def _except_semi_colon_sentence(self, sentences: List[str], start_idx: int, end_idx: int) -> List[str]:
        if sentences[-1][-1].strip() == ":":
            return sentences[:-1], start_idx, end_idx - 1
        return sentences, start_idx, end_idx
    
    
    def _is_not_informative_paragraph(self, sentences: List[str]) -> bool:
        paragraph = " ".join(sentences)
        # return not (0 < len(sentences) < 6 and self.min_threshold < len(paragraph) < self.max_threshold)
        return not (len(sentences) <= 10 and self.min_threshold < len(paragraph))
    
    def _set_sentences_label(self, new_label: Dict, old_label: Dict):
        self.same_label_idx, self.new_label_idx, self.contradict_label_idx = set(), set(), set()
        for idx in new_label["S"]:
            self.same_label_idx.add(int(idx)) # 여기 new_label[id]["S"][idx] == None 인 경우도 있는데 어떻게 처리해야할지 논의 필요
        for idx in new_label["N"]["indices"]:
            self.new_label_idx.add(idx)
        for old_idx, new_idx in old_label["C"]["indices"]:
            self.contradict_label_idx.add(new_idx)
        self.labelled_idx = self.same_label_idx | self.new_label_idx | self.contradict_label_idx
    
    def _is_not_updated_paragraph(self, sentence_type: List) -> bool:
        return dialogue_types["new"] not in sentence_type and dialogue_types["contradict"] not in sentence_type
    
    def _is_unlabelled_paragraph(self, start_idx: int, end_idx: int) -> bool:
        target_sentences = set(range(start_idx, end_idx))
        return len(target_sentences - self.labelled_idx) != 0
    
    def _get_sentences_type(self, start_idx: int, end_idx: int) -> List[str]:
        sentences_type = []
        for idx in range(start_idx, end_idx):
            if idx in self.contradict_label_idx:
                sentences_type.append(dialogue_types["contradict"])
            elif idx in self.new_label_idx:
                sentences_type.append(dialogue_types["new"])
            else:
                sentences_type.append(dialogue_types["same"])
        return sentences_type
    
    
    def _final_candidate_paragraphs(self, contradict_paragraphs: List, new_paragraphs: List) -> List:
        if len(contradict_paragraphs) > 0 or len(new_paragraphs) > 0:
            print("contra: ", len(contradict_paragraphs), "new: ", len(new_paragraphs))
            
        contra_idx = min(len(contradict_paragraphs), self.num_of_contradict_dialogue)
        new_idx = min(len(new_paragraphs), self.num_of_new_dialogue)
        tmp_idx = max(self.num_of_dialogue - contra_idx - new_idx, 0)
        
        tmp_paragraphs = contradict_paragraphs[contra_idx:] + new_paragraphs[new_idx:]
        random.shuffle(contradict_paragraphs)
        random.shuffle(new_paragraphs)
        random.shuffle(tmp_paragraphs)
        
        return contradict_paragraphs[:contra_idx] + new_paragraphs[:new_idx] + tmp_paragraphs[:tmp_idx]
        