import argparse
import pprint
import re
import string
from rouge import Rouge
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
    prediction_tokens = normalize_answer(' '.join(prediction.split()[:10])).split()
    ground_truth_tokens = normalize_answer(' '.join(ground_truth.split()[:10])).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    #if f1 >= 0.5:
    #    return f1
    #else:
    #    return 0
    return f1


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# F1 score definition
def _f1_score_only_tag(prediction, ground_truth, tag=['NNP', 'NN', 'NNS', 'NNPS', 'CD']):
    prediction = normalize_answer(prediction)#.split()
    ground_truth = normalize_answer(ground_truth)#.split()
    prediction_tokens = [i[0] for i in pos_tag(word_tokenize(prediction)) if i[1] in tag]
    ground_truth_tokens = [i[0] for i in pos_tag(word_tokenize(ground_truth)) if i[1] in tag]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    #print(common)
    #time.sleep(1)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    #if f1 >= 0.5:
    #    return f1
    #else:
    #    return 0
    return f1


# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def _get_gold_and_pred(results, eval_type):
    if eval_type == "CHANGED_DEL":
        print('Selected Type is eval_type')
        results = list(filter(lambda x: x[1]['type'] == 'CHANGED', results))
        golds = list(map(lambda x: normalize_answer(x[1]['old_answer']), results))
        preds = list(map(lambda x: normalize_answer(x[1]['prediction']), results))
    else:    
        if eval_type != 'ALL':
            results = list(filter(lambda x: x[1]['type'] == eval_type, results))
            print('Selected Type is', results[0][1]['type'])
        golds = list(map(lambda x: normalize_answer(x[1]['answer']), results))
        preds = list(map(lambda x: normalize_answer(x[1]['prediction']), results))
    return golds, preds


def _calculate_metrics(results, eval_type):
    
    golds, preds = _get_gold_and_pred(results, eval_type)
    total_count = len(golds)

    # downstream metrics
    em = 0
    f1 = 0
    rougel = 0

    for gold, pred in zip(golds, preds):
        if len(pred) == 0: # empty answer
            continue
        if pred==gold:
            em += 1 
        f1 += _f1_score(pred, gold)
        rougel += _rougel_score(pred, gold)

    if total_count > 0:
        em /= total_count
        f1 /= total_count
        rougel /= total_count

    print(round(em*100, 2))
    print(round(f1*100, 2))
    print(round(rougel*100, 2))
    
    return {
        "downstream": {
            "em": em,
            "f1": f1,
            "rougel": rougel,
        },
    }


def evaluate(fn):
    with open(fn) as f:
        js = json.load(f)
        results = list(js.items())
        
    for eval_type in ['ALL', 'SAME', 'NEW', 'CHANGED', 'CHANGED_DEL']:
        result = _calculate_metrics(results, eval_type)





if __name__ == "__main__":
    fire.Fire(evaluate)