import re
import random
from typing import List, Tuple, Dict, Set

dialogue_types = {
    "same": "SAME",
    "new": "NEW",
    "contradict": "CONTRADICT",
    "deleted": "DELETED",
    "unlabelled": "UNLABELLED",
}

def get_empty_dialogue(title: str) -> Dict:
    return {"title": title, "data": []}

def article_id_is_not_consistent(*data_list: Tuple) -> bool:
    return not all([set(data_list[0].keys()) == set(data.keys()) for data in data_list])

def article_id(data: Dict) -> str:
    return list(data.keys())[0]

def article_is_empty(article: str) -> bool:
    return len(article.strip()) == 0

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_paragraph_into_sentences(text):
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(". ",".<stop> ")
    text = text.replace("? ","?<stop> ")
    text = text.replace("! ","!<stop> ")
    text = text.replace(": ",":<stop> ")
    text = text.replace("; ",";<stop> ")
    text = text.replace("et al.<stop>", "et al.")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def split_article_into_sentences(text):
    paragraphs = text.split('\n')
    sentences, paragraph_indices = [], []
    cur_idx = 0
    
    for p in paragraphs:
        sent = split_paragraph_into_sentences(p)
        if len(sent) == 0:
            continue
        if len(sent) == 1 and len(sent[0]) < 50:
            continue
        sentences.extend(sent)
        
        paragraph_indices.append([cur_idx, cur_idx + len(sent)])
        cur_idx += len(sent)
        
    return sentences, paragraph_indices