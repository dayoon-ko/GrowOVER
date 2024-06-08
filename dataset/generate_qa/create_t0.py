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
from utils import seed_everything
from qa_manager import generate_qa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0_month", type=int, default=9)
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

    root = Path(
        None
    )
    files = [
        str(i)
        for i in sorted(root.glob(f"{month:02}/AA/wiki_00.json"))
        if not str(i).endswith("result.json")
    ]
    # files = files[:10]
    # files = files[1]

    with open(files[0]) as f:
        js = json.load(f)

    curr_idx = min([int(i) for i in js])
    results = {}

    while True:
        if len(js[str(curr_idx)][2]) == 0:
            results[str(curr_idx)] = None

        else:
            results[str(curr_idx)] = generate_qa(model, js[str(curr_idx)], num_clusters)

        del js[str(curr_idx)]

        # if js ended, save and load
        if len(js) == 0:
            save_dir = f"{save_root}/{files[0].split('/')[-2]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_fn = save_dir + "/" + files[0].split("/")[-1]
            with open(save_fn, "w") as f:
                json.dump(results, f, indent=3)
                print("Dump :", save_fn)

            del files[0]
            if len(files) == 0:
                break

            with open(files[0]) as f:
                js = json.load(f)
            results = {}

        # update curr_idx
        curr_idx = min([int(i) for i in js])

        ## TODO: erase
        # if curr_idx == 13:
        #     save_dir = f"{save_root}/{files[0].split('/')[-2]}"
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir, exist_ok=True)
        #     save_fn = save_dir + "/" + files[0].split('/')[-1]
        #     with open(save_fn, 'w') as f:
        #         json.dump(results, f, indent=3)
        #         print('Dump :', save_fn)
        #     break


if __name__ == "__main__":
    args = get_args()
    main(args)
