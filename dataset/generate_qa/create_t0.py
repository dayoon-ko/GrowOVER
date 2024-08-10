import torch
import json
import re
import argparse
import openai
import os
import random
import numpy as np

from simcse import SimCSE
from tqdm.auto import tqdm
from pathlib import Path
from utils import seed_everything, load_json
from qa_manager import generate_qa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0_month", type=int, default=8)
    parser.add_argument("--wiki_dump_root", type=str)
    parser.add_argument("--save_root", type=str, default="QA_result")
    parser.add_argument("--n_clusters", type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    seed_everything()

    args = get_args()
    month = args.t0_month
    save_root = args.save_root + f"/{month:02d}"
    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    num_clusters = args.n_clusters

    files = [str(i) for i in sorted(Path(args.wiki_dump_root).glob(f"{month:02}/**/wiki*"))]
    curr_file = load_json(files[0])

    curr_idx = min([int(i) for i in curr_file])
    results = {}

    while True:
        if len(curr_file[str(curr_idx)]["text"]) == 0:
            results[str(curr_idx)] = None
        else:
            results[str(curr_idx)] = generate_qa(model, curr_file[str(curr_idx)], num_clusters)
        del curr_file[str(curr_idx)]

        # if js ended, save and load
        if len(curr_file) == 0:
            
            save_dir = f"{save_root}/{files[0].split('/')[-2]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            save_fn = save_dir + "/" + files[0].split("/")[-1]
            with open(save_fn, "w") as f:
                json.dump(results, f, indent=2)
                print("Dump :", save_fn)

            del files[0]
            if len(files) == 0:
                break

            curr_file = load_json(files[0])
            results = {}

        # update curr_idx
        curr_idx = min([int(i) for i in curr_file])


if __name__ == "__main__":
    args = get_args()
    main(args)
