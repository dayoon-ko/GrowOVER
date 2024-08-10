import re
from typing import Dict, List, Tuple

prompt = """Create an Information Dialogue Dataset about {topic} between two conversation partners (User, Expert).
A paragraph about {topic} will be provided as factual information. The expert's words must be generated to provide an answer based on this information.

Using the following instruction for generating a dialogue:
1) The user starts the dialogue first
2) Create a multi-turn dialogue of 3-4 turns, each consisting of a not too long conversation.
3) Create it to include each element of conversation, discussion, and QA. In other words, users should not always ask questions using interrogative sentences.
4) DON'T use phrases such as according to the paragraph in guide's utterance.
5) DON'T simply parrot this paragraph or referenced directly. There is no need to include everything given in the paragraph in the dialogue.
6) Do not use what you already know about {topic}, and the Expert will answer only with the content of the provided paragraph.
7) I will provide you with sentences and a unique number for each sentence. You must indicate the Sentence number you've referenced for each turn.

Below is an example of output format and dialogues:
{{Reference Sentence}}2{{User}}I really love Granny Smith apples, they’re my favorite type of apple{{Expert}}I love granny smith apples. they have hard, light green skin and a crisp flesh.
{{Reference Sentence}}1{{User}}Yes, I really enjoy them. I also like Honeycrisp apples but they’re so expensive!{{Expert}}they’ve been grown for thousands of years in asia and europe, and were brought to north america by european colonists
{{Reference Sentence}}3{{User}}Oh really? They’ve been around way longer than I thought!{{Expert}}they’re also consumed raw, it’s one of the most popular cooking apples.

Sentences:
{sentences}

Please generate dialogue:
"""

def get_prompt(args: dict) -> str:
    return prompt.format(**args)


class DialogueParser:
    def parse_dialogue(self, response: str, start_index: int, paragraph: List[str], type: List[str]) -> Dict:
        """
            {Reference Sentence}1{User}User's utterance{Expert}Expert's utterance
            {Reference Sentence}3{User}User's utterance{Expert}Expert's utterance
            {Reference Sentence}2{User}User's utterance{Expert}Expert's utterance
        """
        dialogue = []
        while True:
            reference_sentence_match = re.search(r"{Reference Sentence}(\d+){", response)
            user_match = re.search(r'{User}([^}]+){', response)
            expert_match = re.search(r'{Expert}([^}]+){', response)
            
            if reference_sentence_match is None or user_match is None:
                break
            if expert_match is None:
                expert_match = re.search(r'{Expert}([^}]+)', response)
            
            sentence_index = int(reference_sentence_match.group(1).strip()) - 1
            user = user_match.group(1).strip()
            expert = expert_match.group(1).strip()
            
            try:
                paragraph[sentence_index]
            except:
                print(sentence_index, len(paragraph))
                print(paragraph)
                return {"dialogue": []}
            
            dialogue.append({
                "user": user, 
                "expert": expert,
                "sentence_type": type[sentence_index], 
                "sentence_index": str(sentence_index + start_index),
                "grounded_sentence": paragraph[sentence_index],
                })
            response = response[expert_match.end() - 1:]
                
        return {
            "dialogue": dialogue,
        }