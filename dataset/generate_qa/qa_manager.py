import re
import random
from tqdm.auto import tqdm
from clustering import adjacent_cluster_sequence, cluster_sequence
from utils import split_article_into_sentences, split_paragraph_and_sentence
from qa_prompt import t0_prompt, tn_prompt, tn_with_surrounding_prompt


qa_types = ["SAME", "NEW", "CHANGED_NEW", "DELETED", "CHANGED_DELETED"]


def get_bounding_box(lst):
    """
    Given list of indices, get bounding box.
    """
    indices = []
    for item in lst:
        if isinstance(item, list):
            indices.extend(list(range(item[0], item[1])))
        else:
            indices.append(item)

    return min(indices), max(indices) + 1


def tie_indices(model, sent, lst):
    """
    If bounding box is too large, tie indices using adjacent clustering
    """
    consecutive_tied = [[lst[0]]]

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] == 1:
            consecutive_tied[-1].append(lst[i])
        else:
            consecutive_tied.append([lst[i]])

    result = []

    for lst in consecutive_tied:
        if len(lst) >= 6:
            out = adjacent_cluster_sequence(model, len(lst) // 3, sent, lst)
            for clustered_lst in out:
                result.append((min(clustered_lst), max(clustered_lst) + 1))
        else:
            result.append((min(lst), max(lst) + 1))

    return result


def get_surrounding_sentences(sentence_list, start, end):
    """
    Given sentence_list, start index, end index (of sentences),
    return two sentences before and after the corresponding sentence(s).
    """
    start_context = max(0, start - 2)
    end_context = min(len(sentence_list), end + 2)

    relative_start = start - start_context
    relative_end = end - start_context

    surrounding_sentences = sentence_list[start_context:end_context]
    focus_sentences = sentence_list[start:end]

    return (
        surrounding_sentences,
        focus_sentences,
        list(range(relative_start, relative_end)),
    )


def extract_context_sentences_with_relative_indices(
    sentence_list, start_index, end_index
):
    # Check if start_index and end_index are within the range of sentence_list
    if start_index < 0 or end_index > len(sentence_list):
        return "Invalid start or end index."

    # Calculate the indices for two sentences before and after
    start_context = max(0, start_index - 2)
    end_context = min(len(sentence_list), end_index + 2)

    # Extract the sentences
    before_sentences = sentence_list[start_context:start_index]
    after_sentences = sentence_list[end_index:end_context]

    # Calculate the relative indices of the original sentences within the context
    relative_start_index = start_index - start_context
    relative_end_index = end_index - start_context

    return before_sentences, after_sentences, (relative_start_index, relative_end_index)


def generate_qa(model, wiki, num_clusters):
    """
    Initialize QA pairs.
    Cluster paragraphs and randomly select paragraphs for each cluster.
    """
    out = []
    title = wiki["title"]
    article = wiki["text"]
    sent, indices = split_paragraph_and_sentence(article)
    items = cluster_sequence(model, num_clusters, sent, indices)

    for idx in tqdm(items):
        paragraph = [sent[i] for i in range(*idx)]
        prompt_result = t0_prompt(title, paragraph)
        extracted_list = re.findall(r"\{(.*?)\}", prompt_result)
        res = {}

        if len(extracted_list) != 4:
            continue
        try:
            start_idx = idx[0] + int(extracted_list[2])

            if int(extracted_list[3]) >= (idx[1] - idx[0]):
                end_idx = idx[1]
            else:
                end_idx = idx[0] + int(extracted_list[3]) + 1

        except:
            print("The string cannot be converted to an integer")
            continue
        if end_idx - start_idx < 1:
            continue

        res["type"] = "NEW"
        res["question"] = extracted_list[0]
        res["answer"] = extracted_list[1]
        res["grounded_text"] = " ".join([sent[i] for i in range(start_idx, end_idx)])
        res["start_idx"] = start_idx
        res["end_idx"] = end_idx
        out.append(res)

    out = sorted(out, key=lambda x: x["start_idx"])

    if len(out) == 0:
        return None
    return out


def update_qa(model, label_old, label_new, old_qa, new_wiki):
    """
    Update QA pairs.
    1) Maintain or delete existing QA pairs.
    2) Create changed / new QA pairs.
    """
    new_qa = []
    title = new_wiki[1]
    article = new_wiki[2]
    sentence_list = split_article_into_sentences(article)

    # Maintaining or Deleting old_qa pairs
    for qa_item in old_qa:
        if qa_item["type"] in qa_types[3:]:  # DELETED / CHANGED_DELETED
            continue  # don't add in new_qa

        matched_indices = []

        # if there is partial new, abandon qa
        is_partial_new = False

        for _, value in label_new["N"].items():
            if value in list(range(qa_item["start_idx"], qa_item["end_idx"])):
                is_partial_new = True

        if is_partial_new:
            continue

        # old_label types:
        # S --- match_index
        #   |-- match_range

        # S2 -- match_index
        #   |-- match_range?

        # D --- X           -> -3
        #   |-- match_index -> -1

        # C --- match_index -> -2
        for old_idx in range(qa_item["start_idx"], qa_item["end_idx"]):
            idx_key = str(old_idx)
            if idx_key in label_old["C"]:
                matched_indices.append("CHANGE-DELETE")  # -2
            elif idx_key in label_old["D"]:
                type = (
                    "DELETE" if label_old["D"][idx_key] is None else "ABANDON"
                )  # -3 / -1
                matched_indices.append(type)
            elif idx_key in label_old["S"]:
                matched_indices.append(label_old["S"][idx_key])
            elif idx_key in label_old["S2"]:
                matched_indices.append(label_old["S2"][idx_key])
            else:
                print("old idx doesn't belong to any types: ", old_idx)
                matched_indices.append(-1)

        if -1 in matched_indices:
            continue
        if "ABANDON" in matched_indices:
            continue
        elif "CHANGE-DELETE" in matched_indices:
            qa_item["type"] = qa_types[4]  # 'CHANGED_DELETED'
            qa_item["start_idx"] = -1
            qa_item["end_idx"] = -1
        elif "DELETE" in matched_indices:
            qa_item["type"] = qa_types[3]  # 'DELETED'
            qa_item["start_idx"] = -1
            qa_item["end_idx"] = -1
        else:
            qa_item["type"] = qa_types[0]
            qa_item["start_idx"], qa_item["end_idx"] = get_bounding_box(matched_indices)
            qa_item["grounded_text"] = " ".join(
                sentence_list[qa_item["start_idx"] : qa_item["end_idx"]]
            )

        new_qa.append(qa_item)

    # Creating Changed / New QA pairs

    new_indices = sorted(
        list(set([int(key) for key, value in label_new["N"].items() if value is None]))
    )
    changed_indices = sorted(
        list(set([value for _, value in label_old["C"].items() if value is not None]))
    )

    if len(new_indices) != 0:
        new_indices = tie_indices(model, sentence_list, new_indices)
        if len(new_indices) > 10:
            new_indices = random.sample(new_indices, 10)
        for start, end in new_indices:
            if end - start <= 2:
                surrounding_sentence, sentence, indicator = get_surrounding_sentences(
                    sentence_list, start, end
                )
                res = tn_with_surrounding_prompt(title, surrounding_sentence, indicator)
            else:
                sentence = sentence_list[start:end]
                res = tn_prompt(title, sentence)

            extracted_list = re.findall(r"\{(.*?)\}", res)
            res = {}

            if len(extracted_list) != 2:
                continue

            res["type"] = qa_types[1]  # 'NEW'
            res["question"] = extracted_list[0]
            res["answer"] = extracted_list[1]
            res["grounded_text"] = " ".join(sentence)
            res["start_idx"] = start
            res["end_idx"] = end
            new_qa.append(res)

    if len(changed_indices) != 0:
        changed_indices = tie_indices(model, sentence_list, changed_indices)
        if len(changed_indices) > 10:
            changed_indices = random.sample(changed_indices, 10)

        for start, end in changed_indices:
            if end - start <= 2:
                surrounding_sentence, sentence, indicator = get_surrounding_sentences(
                    sentence_list, start, end
                )
                res = tn_with_surrounding_prompt(title, surrounding_sentence, indicator)
            else:
                sentence = sentence_list[start:end]
                res = tn_prompt(title, sentence)

            extracted_list = re.findall(r"\{(.*?)\}", res)
            res = {}

            if len(extracted_list) != 2:
                continue

            res["type"] = qa_types[2]  # 'CHANGED_NEW'
            res["question"] = extracted_list[0]
            res["answer"] = extracted_list[1]
            res["grounded_text"] = " ".join(sentence)
            res["start_idx"] = start
            res["end_idx"] = end
            new_qa.append(res)

    new_qa = sorted(new_qa, key=lambda x: x["start_idx"])

    if len(new_qa) == 0:
        return None
    return new_qa


def mark_deleted(old_qa):
    new_qa = []
    for qa_item in old_qa:
        if qa_item["type"] in qa_types[3:]:  # DELETED / CHANGED_DELETED
            continue  # don't add in new_qa

        qa_item["type"] = qa_types[3]
        qa_item["start_idx"] = -1
        qa_item["end_idx"] = -1

        new_qa.append(qa_item)

    if len(new_qa) == 0:
        return None
    return new_qa
