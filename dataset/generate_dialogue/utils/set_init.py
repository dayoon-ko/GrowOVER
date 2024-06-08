import os
import openai
from argparse import ArgumentParser, ArgumentTypeError
from typing import Dict
from dotenv import load_dotenv

from retrievals import retrieval_classes

# load .env
load_dotenv()

def init_setting():
    init_openai_api_key()
    

def init_openai_api_key():
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_args(defaults: dict = None):
    
    #--- add argument ---#
    parser = ArgumentParser()
    # parser.add_argument('--data_root', default=data_root, type=str, help='root directory of data')
    # parser.add_argument('--retrieval', default="simcse", type=str, help='retrieval method, you can choose from {}'.format(list(retrieval_classes.keys())))

    #--- add default argument ---#
    add_dict_to_argparser(parser, defaults)

    #--- create args ---#
    args = parser.parse_args()
    
    return args


def add_dict_to_argparser(parser: ArgumentParser, default: Dict):
    for k, v in default.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
        

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("boolean value expected")