import os
import logging
import openai
import random
from datetime import datetime
from dotenv import load_dotenv
from argparse import ArgumentParser, Namespace

load_dotenv()

def init_settings():    
    set_openai_api_key()
    set_seed()
    set_logger()


def set_openai_api_key():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    

def set_seed(seed:int = 42):
    random.seed(seed)
    
def set_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"logs/{now}.log", 
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    
def get_args() -> Namespace:
    parser = ArgumentParser()
    
    #--- directory ---# 
    parser.add_argument("--wiki-dir", default=os.getenv("WIKI_DIR"), type=str, help="root directory of wikipedia snapshot")
    parser.add_argument("--label-dir", default=os.getenv("LABEL_DIR"), type=str, help="root directory of updated labels for wikipedia snapshot")
    parser.add_argument("--dialogue-dir", default=os.getenv("DIALOGUE_DIR"), type=str, help="save directory of dialogues")
    parser.add_argument("--month", type=int, help="month for updating dialogues")
    
    #--- openai ---#
    parser.add_argument("--model", default="gpt-4-1106-preview", type=str, help="openai model")
    parser.add_argument("--temperature", default=0.0, type=float, help="openai temperature")
    parser.add_argument("--max-tokens", default=1024, type=int, help="openai max tokens")
    
    #--- dialogue ---#
    parser.add_argument("--num-of-dialogue", default=4, type=int, help="number of mult-turn dialogue set per topic.")
    parser.add_argument("--num-of-new-dialogue", default=4, type=int, help="number of new dialogue per topic.")
    parser.add_argument("--num-of-contradict-dialogue", default=4, type=int, help="number of contradict dialogue per topic.")
    parser.add_argument("--min-threshold", default=350, type=int, help="minimum threshold of paragraph length")
    parser.add_argument("--max-threshold", default=1000, type=int, help="maximum threshold of paragraph length")
    
    args = parser.parse_args()
    return args