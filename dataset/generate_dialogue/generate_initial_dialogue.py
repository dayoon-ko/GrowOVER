from tqdm import tqdm
from argparse import Namespace
from typing import Dict, List

from utils.set_init import init_setting, get_args
from utils.data_loader import DataLoader
from utils.data_saver import DataSaver
from utils.commons import text_is_empty, get_empty_dialogue, run_prompt, dialogue_is_empty
from prompts.initial_dialogue import get_prompt, parse_dialogue

"""
    This code is used to generate initial dialogue for each starting topic(wikipedia title).
    Result is some pair of (guide, explore) dialogue for each starting topic.
"""
    
def main(args: Namespace):
    # set environment variables
    init_setting()
    
    data_loader = DataLoader(args.data_root)
    data_saver = DataSaver(args.save_dir)
    
    # load each wikipedia json file and generate initial dialogue
    for data, path in tqdm(data_loader):
        initial_dialogues = {}
        data_indices = data.keys()
        
        for data_idx in tqdm(data_indices):
            url, title, text = data[data_idx]
            
            #--- generate initial dialogue ---#
            if text_is_empty(text):
                dialogue = get_empty_dialogue(data_idx) # empty dialogue
            else:
                dialogue = generate_initial_dialogue(topic=title, num_dialogue=args.num_dialogue) # generate initial dialogue
            
            #--- save generated dialogue ---#
            initial_dialogues[data_idx] = dialogue
            
        #--- save initial dialogue ---#
        data_saver.save(initial_dialogues, path, args.data_root)



def generate_initial_dialogue(topic: str, num_dialogue: int = 1) -> List:
    """
        generate initial dialogue for given topic
        :param topic: starting topic
        :param num_dialogue: number of dialogue to generate
        :return: list of dialogue
    """
    dialogue = []
    prompt = get_prompt(topic=topic)
    
    for _ in range(num_dialogue):
        _, generated_dialogue = run_prompt(prompt, temperature=0.7)
        parsed_dialogue = parse_dialogue(generated_dialogue)
        
        if dialogue_is_empty(parsed_dialogue):
            continue
        dialogue.append(parsed_dialogue)
    
    return dialogue
    
    

if __name__ == "__main__":
    defaults = dict(
        data_root = "TemporalWikiDatasets/Wikipedia_datasets/08",
        save_dir = "dialogue/initial_dialogue",
        num_dialogue = 1,
    )
    args = get_args(defaults=defaults)
    
    main(args)