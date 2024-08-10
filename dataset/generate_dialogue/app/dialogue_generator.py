import openai
import logging
from argparse import Namespace
from typing import List, Dict

from app.prompt.generate_dialogue import get_prompt, DialogueParser
from app.utils import dialogue_types


class DialogueGenerator:
    def __init__(self, args: Namespace):
        self.month = args.month
        
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        
        self.dialogue_parser = DialogueParser()

    
    def run(self, id: str, title: str, target_paragraphs: List) -> Dict:
        """
            Args:
                id: str
                title: str
                target_paragraphs: List (# sentences(List), start_index(int), type(str))
        """

        generated_data, generated_dialogue = [], dict()

        for paragraph, start_index, type in target_paragraphs:
            logging.info(f"Generating dialogue in article {id} (title: {title}) {{{len(generated_data) + 1} / {len(target_paragraphs)}}}")
            try:
                generated_dialogue = self._generate_dialogue(title, paragraph)
            except Exception as e:
                logging.warning(f"Failed to generate dialogue in article {id} (title: {title}) with target paragraph {paragraph}")
                logging.warning(e)
            
            dialogue = self.dialogue_parser.parse_dialogue(generated_dialogue, start_index, paragraph, type)
            if dialogue["dialogue"] == []:
                logging.warning(f"Failed to parse dialogue in article {id} (title: {title}) with generated dialogue {generated_dialogue}")
                continue
                
            dialogue.update({
                "article_id": id,
                "created_month": self.month, 
                "last_modified_month": self.month,
                "dialogue_type": dialogue_types["new"],
                })
            
            generated_data.append(dialogue)
            
        return generated_data
    
    
    def _generate_dialogue(self, title: str, paragraph: List) -> Dict:
        prompt_args = self._get_prompt_args(title, paragraph)
        prompt = get_prompt(prompt_args)
        response = self._run_prompt(prompt)
        return response
    
    
    # TODO: check after update prompt
    def _get_prompt_args(self, title: str, paragraph: List[str]) -> Dict:
        sentences = "\n\n".join([f"Sentence {i+1}\n {sentence}" for i, sentence in enumerate(paragraph)])
        prompt_args = {
            "topic": title,
            "sentences": sentences,
        }
        return prompt_args  
    
    
    def _run_prompt(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = [
            {"role" : "system", "content" : "You are a helpful assistant."},
            {"role" : "user", "content" : prompt}
            ],
            temperature = self.temperature,
            max_tokens = self.max_tokens
        )
        return response['choices'][0]['message']['content']