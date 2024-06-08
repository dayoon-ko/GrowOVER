from typing import Dict

from utils.commons import text_is_empty, normalize_format, get_valid_dialogue, get_empty_dialogue

generate_initial_dialogue_prompt = '''
Please Create a dialogue between two individuals (Guide, Explorer) demanding specialized knowledge about {topic}.

In this scenario, the two individuals possess asymmetric knowledge about {topic}.
The Guide is an expert in {topic}, while the Explorer has no prior knowledge about {topic}.

The dialogue should start with the Guide presenting intriguing information about {topic}, followed by the Explorer asking questions about aspects they find curious or do not comprehend.
KEEP IN MIND that the Explorer is in a situation where they are learning about {topic} for the first time.
So the Guide should not assume that the Explorer has any prior knowledge about {topic}.

Dialogue consists of just ONE conversation with the words of the Guide and questions from the Explorer. 
The Guide DOSE NOT NEED to respond to the Explorer's questions.

Below is an example of the output format:

"Guide": "Start comment"
"Explorer": "Question"

Please generate a dialogue
'''

def get_prompt(topic: str) -> str:
    # return prompt for given topic
    return generate_initial_dialogue_prompt.format(topic=topic).strip()



def parse_dialogue(generated_dialogue: str) -> Dict[str, str]:
    """
        parse generated dialogue
        :param generated_dialogue: generated dialogue
        :return: parsed dialogue
    """
    guide, explorer = "Guide:", "Explorer:"

    guide_to_normalize = ["Guide:", "Guide :", "guide:", "guide :"]
    explorer_to_normalize = ["Explorer:", "Explorer :", "explorer:", "explorer :"]
    
    # normalize format for parsing
    normalized_dialogue = normalize_format(generated_dialogue, guide_to_normalize, guide)
    normalized_dialogue = normalize_format(normalized_dialogue, explorer_to_normalize, explorer)
    
    # split dialogue into (guide, explore, guide, explore, ...) -> (guide, explore), (guide, explore), ... 
    splited_dialogues = normalized_dialogue.split(guide)
    
    # get valid initial dialogue
    initial_dialogue = get_valid_dialogue(splited_dialogues, delimiter=explorer)
    
    if text_is_empty(initial_dialogue):
        return get_empty_dialogue()
    
    guide, explore = initial_dialogue.split(explorer)
    guide, explore = guide.strip(), explore.strip()

    return {"guide": guide, "explore": explore}

