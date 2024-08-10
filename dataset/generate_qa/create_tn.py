import torch
import json
import argparse
import os
from pathlib import Path
from simcse import SimCSE
from qa_manager import generate_qa, update_qa, mark_deleted


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month_old", type=int, default=9)
    parser.add_argument("--wiki_dump_root", type=str)
    parser.add_argument("--label_new_root", type=str)
    parser.add_argument("--label_chg_root", type=str)
    parser.add_argument("--save_root", type=str, default="QA_result")
    parser.add_argument("--n_clusters", type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    args = get_args()
    month_old = args.month_old
    month_new = month_old + 1
    save_root = args.save_root + f"/{month_new:02d}"
    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    num_clusters = args.n_clusters

    qa_root = Path("QA/" + args.save_root)
    qa_old = [str(i) for i in sorted(qa_root.glob(f"{month_old:02}/*/wiki_*.json"))]
    wiki_files_new = [str(i) for i in sorted(root.glob(f"{month_new:02}/*/wiki_*.json"))]
    label_old = [str(i) for i in sorted(
                    label_chg_root.glob(f"{month_old:02}{month_new:02}/{month_old:02}/*/wiki*")
                )]
    label_new = [str(i) for i in sorted(
                    label_new_root.glob(f"{month_old:02}{month_new:02}/{month_new:02}/*/wiki*")
                )]
    
    # Input Files:
    ## 1. QA result : Old Month = qa_old
    ## 2. Wikipedia context : New Month = files_new
    ## 3. Labeling result : 0809/08 + 0809/09

    with open(qa_old[0]) as f:
        old_qa = json.load(f)
    with open(files_new[0]) as f:
        new_wiki = json.load(f)

    with open(label_old[0]) as f:
        label_old = json.load(f)
    with open(label_new[0]) as f:
        label_new = json.load(f)

    old_idx = min([int(i) for i in old_qa])
    new_idx = min([int(i) for i in new_wiki])
    # max_idx_old = max([int(i) for i in old_qa])
    # max_idx_new = max([int(i) for i in new_wiki])
    max_idx_label_old = max([int(i) for i in label_old])
    max_idx_label_new = max([int(i) for i in label_new])

    results_new = {}

    while True:
        # if old js ended, load next old js
        if len(old_qa) == 0:
            del qa_old[0]

            with open(qa_old[0]) as f:
                old_qa = json.load(f)

        # if new js ended, save and load next new js
        if len(new_wiki) == 0:
            save_dir = f"{save_root}/{files_new[0].split('/')[-2]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_fn = save_dir + "/" + files_new[0].split("/")[-1]
            with open(save_fn, "w") as f:
                json.dump(results_new, f, indent=3)
                print("Dump : ", save_fn)

            del files_new[0]

            # break if all files are iterated
            if len(files_new) == 0:
                break

            with open(files_new[0]) as f:
                new_wiki = json.load(f)
            results_new = {}

        old_idx = min([int(i) for i in old_qa])
        new_idx = min([int(i) for i in new_wiki])

        cur_idx = min([old_idx, new_idx])

        # load label files if cur_idx is higher than max_idx
        if cur_idx > max_idx_label_old:
            del label_old[0]
            with open(label_old[0]) as f:
                label_old = json.load(f)
            max_idx_label_old = max([int(i) for i in label_old])

        if cur_idx > max_idx_label_new:
            del label_new[0]
            with open(label_new[0]) as f:
                label_new = json.load(f)
            max_idx_label_new = max([int(i) for i in label_new])

        # key not in new
        if new_idx > cur_idx:
            results_new[str(cur_idx)] = (
                None
                if old_qa[str(cur_idx)] is None
                else mark_deleted(old_qa[str(cur_idx)])
            )
            del old_qa[str(cur_idx)]

        # key not in old
        elif old_idx > cur_idx:
            results_new[str(cur_idx)] = (
                None
                if new_wiki[str(cur_idx)] is None
                else generate_qa(model, new_wiki[str(cur_idx)], num_clusters)
            )

            del new_wiki[str(cur_idx)]

        # else old_idx = new_idx = cur_idx
        else:
            is_null_old_qa = old_qa[str(cur_idx)] is None
            is_null_new_wiki = len(new_wiki[str(cur_idx)][2]) == 0

            if is_null_old_qa and is_null_new_wiki:
                results_new[str(cur_idx)] = None

            # if old qa is None, initialize qa
            elif is_null_old_qa:
                results_new[str(cur_idx)] = (
                    None
                    if new_wiki[str(cur_idx)] is None
                    else generate_qa(model, new_wiki[str(cur_idx)], num_clusters)
                )

            # if new wiki is None, mark every qa as deleted
            elif is_null_new_wiki:
                results_new[str(cur_idx)] = (
                    None
                    if old_qa[str(cur_idx)] is None
                    else mark_deleted(old_qa[str(cur_idx)])
                )

            else:
                results_new[str(cur_idx)] = update_qa(
                    model,
                    label_old[str(cur_idx)],
                    label_new[str(cur_idx)],
                    old_qa[str(cur_idx)],
                    new_wiki[str(cur_idx)],
                )

                save_dir = f"{save_root}/{files_new[0].split('/')[-2]}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                save_fn = save_dir + "/" + files_new[0].split("/")[-1]
                with open(save_fn, "w") as f:
                    json.dump(results_new, f, indent=3)
                    print("Dump until: ", str(cur_idx))

            del old_qa[str(cur_idx)]
            del new_wiki[str(cur_idx)]


if __name__ == "__main__":
    args = get_args()
    main(args)
