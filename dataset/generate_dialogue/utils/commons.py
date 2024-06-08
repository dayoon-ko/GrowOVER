import openai
from typing import Dict, Tuple, Union, List


def text_is_empty(text: str) -> bool:
    # check if text is empty string or not
    return len(text.strip()) == 0


def get_empty_dialogue(data_idx: str="") -> Union[Dict, None]:
    # return empty dialogue 
    # 우선은 Null로 처리
    return None


def dialogue_is_empty(dialogue: Dict) -> bool:
    # check if dialogue is empty or not
    return not dialogue


def contains_word(target_string: str, search_word: str) -> bool:
    return search_word in target_string


def normalize_format(input_string: str, target_words: List, replace_word: str) -> str:
    # replace target_word to replace_word
    normalized_string = input_string
    for word in target_words:
        normalized_string = normalized_string.replace(word, replace_word)
        
    return normalized_string


def get_valid_dialogue(dialogue_list: List, delimiter: str="") -> str:
    valid_dialogue = ""
    
    for dialogue in dialogue_list:
        # remove empty dialogue and unvalid dialogue
        if text_is_empty(dialogue) or not contains_word(dialogue, delimiter):
            continue
        
        # get valid dialogue
        valid_dialogue = dialogue
        break
    
    return valid_dialogue


# 이 부분 상위 경로 utils에 있는 code
def run_prompt(prompt: str, temperature: float=0.0) -> Tuple[str, str]:
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {"role" : "system", "content" : "You are a helpful assistant."},
            {"role" : "user", "content" : prompt}
            ],
        temperature = temperature,
        max_tokens = 1024
    )
    return prompt, response['choices'][0]['message']['content']