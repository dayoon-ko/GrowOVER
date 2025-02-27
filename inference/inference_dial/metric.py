import argparse
import pprint
import re
import string
import fire
import json
from collections import Counter 

# utility to get gold answers
def normalize_answer(answer):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def truncate(text):
        return text.replace('</s>', '').strip().split('\n')[0]
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(truncate(answer)))))



# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(' '.join(prediction.split()[:50])).split()
    ground_truth_tokens = normalize_answer(' '.join(ground_truth.split()[:50])).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _get_gold_and_pred(results, eval_type):
    if eval_type == "CHANGED_DEL":
        results = list(filter(lambda x: x[1]['sentence_type'] == 'CHANGED', results))
        golds = list(map(lambda x: normalize_answer(x[1]['old_answer']), results))
        preds = list(map(lambda x: normalize_answer(x[1]['prediction']['text']), results))
    else:    
        if eval_type != 'ALL':
            results = list(filter(lambda x: x[1]['sentence_type'] == eval_type, results))
        golds = list(map(lambda x: normalize_answer(x[1]['expert']), results))
        preds = list(map(lambda x: normalize_answer(x[1]['prediction']['text']), results))
    return golds, preds


def _calculate_metrics(results, eval_type):
    
    golds, preds = _get_gold_and_pred(results, eval_type)
    total_count = len(golds)

    # downstream metrics
    em = 0
    f1 = 0

    for gold, pred in zip(golds, preds):
        if len(pred) == 0: # empty answer
            continue
        if pred==gold:
            em += 1 
        f1 += _f1_score(pred, gold)

    if total_count > 0:
        em /= total_count
        f1 /= total_count

    print("EM:", round(em*100, 2))
    print("F1:", round(f1*100, 2))
    
    return {
        "downstream": {
            "em": em,
            "f1": f1
        },
    }


def evaluate(fn):
    with open(fn) as f:
        results = json.load(f)
        
    for eval_type in ['ALL', 'SAME', 'NEW', 'CHANGED', 'CHANGED_DEL']:
        print("Eval:", eval_type)
        result = _calculate_metrics(results, eval_type)
        print("---------")


if __name__ == "__main__":
    fire.Fire(evaluate)