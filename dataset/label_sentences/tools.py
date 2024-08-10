import re
import os
import json
import torch
import openai
from tqdm import tqdm
import numpy as np
from pathlib import Path
from simcse import SimCSE
from nltk import sent_tokenize
from fairseq.data.data_utils import collate_tokens

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
    sentences = []
    for p in paragraphs:
        p = split_paragraph_into_sentences(p)
        if len(p) == 0:
            continue
        if len(p) == 1 and len(p[0]) < 50:
            continue
        sentences.extend(p)
    return sentences
    
    
def list_to_dict(res):
    '''Change format of the result'''
    new_s = {}
    for old_idx, new_idx in zip(res['indices'], res['match_indices']):
        if type(new_idx) == int:
            new_s[old_idx] = new_idx
        elif type(new_idx) == list:
            # select the list with minimum length
            min_len, min_idx = new_idx[0][1] - new_idx[0][0], 0
            for i, (s, e) in enumerate(new_idx):
                if e-s < min_len:
                    min_len = e-s
                    min_idx = i
            new_s[old_idx] = new_idx[min_idx]
    return new_s


def load_json(file_name):
    try:
        return json.load(open(file_name))
    except:
        json_file = [json.loads(i) for i in open(file_name).readlines()]
        json_file = sorted(json_file, key=lambda x: int(x["id"]))
        json_dict = {}
        for item in json_file:
            json_dict[item["id"]] = item
        return json_dict


def match_article(model, old_text, new_text, thrs = 0.99):     
    '''Find same sentences and return results of old_text and new_text'''
    comp = Comparator(model, old_text, new_text)
    
    # return None if no text
    if len(comp.doc1.sentences) == 0 or len(comp.doc2.sentences) == 0:
        return None, None
    
    # label same senteces & paragraphs and get the results
    comp.label_same(thrs)
    result_old, result_new = comp.get_label_same_result()    
    return result_old, result_new
        

def save_json(save_root, month, file_list, results, new=True):
    '''save intermediate results'''
    # set dir
    save_dir = f"{save_root}/{month:02d}/{file_list[0].split('/')[-2]}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # save json
    save_fn = save_dir + "/" + file_list[0].split('/')[-1]
    with open(save_fn, 'w') as f:
        json.dump(results, f, indent=2)



def save_and_load_json(save_root, month, file_list, results, new=True):    
    '''save intermediate results and load next input'''
    # save json file
    save_json(save_root, month, file_list, results, new=True)
    if new:
        print('Dump new :', '/'.join(file_list[0].split('/')[-2:]))
    else:
        print('Dump old :', '/'.join(file_list[0].split('/')[-2:]))
    
    if len(file_list) == 1:
        return None, None, None

    del file_list[0]
    
    js = load_json(file_list[0])
    max_idx = max([int(i) for i in js])    
    
    return file_list, js, max_idx
    
    
def init(root, save_root, month, new=True):
    
    input_files = [str(i) for i in sorted(Path(root).glob(f'{month:02}/*/wiki_*'))]
    result_files = [str(i) for i in sorted(Path(save_root).glob(f'{month:02}/*/wiki_*'))]

    # if no saved results, return only input files
    if len(result_files) == 0:
        input_json = load_json(input_files[0])
        return {}, input_files, input_json
    # Otherwise, load the last result and start with the corresponding input files 
    else:
        start_idx = len(result_files) - 1
        input_files = input_files[start_idx:]
        input_json = load_json(input_files[0])
        result_json = load_json(result_files[-1])
        if new:
            for k in result_json:
                del result_json[k]
        return result_json, input_files, input_json


class Document:
    
    def __init__(self, text, title=None):
        self.title = title if title else None
        self.fulltext = text
        self.sentences = split_article_into_sentences(text)
        self.labels = [0 for s in self.sentences] 
        self.match_indices = [-1 for s in self.sentences] # -1 is None
        
    def set_info(self, curr_idx, label, prev_idx):
        self.labels[curr_idx] = label
        self.match_indices[curr_idx] = prev_idx
    
    def get_info(self, idx):
        return {'curr_sent' : self.sentences[idx], 
                'label' : self.labels[idx], 
                'match_index' : self.match_indices[idx]
        }
    
    def get_sentences_from_label(self, label, return_chunks=False):
        if label not in [0, 'S', 'N', 'C', 'D']: 
            return None
        if not return_chunks:
            return {k:s for k, s in enumerate(self.sentences) if self.labels[k] == label}
        if return_chunks:
            indices = []
            match_indices = []
            for i, match_idx in enumerate(self.match_indices):
                if match_idx == -1:
                    continue
                indices.append(i)
                match_indices.append(match_idx)
            return {'indices': indices, 'match_indices': match_indices}
    
    def set_match_indices(self, similarity, thrs=0.98):
        # curr : similarity <- model.similarity(prev, curr)
        # prev : similarity <- model.similarity(prev, curr).transpose(1, 0)
        sim_max = similarity.max(0)
        sim_argmax = similarity.argmax(0)
        sim_max_one = np.where(sim_max >= thrs, 1, -1)
        sim_indices = (sim_max_one * (sim_argmax + 1)) # add 1
        sim_indices = sim_indices - 1 # then subtract 1 
        sim_indices = np.where(sim_indices >= 0, sim_indices, -1)
        self.match_indices = sim_indices.tolist()

    def set_same_labels(self, prev_match_indices):
        for ci, mi in enumerate(self.match_indices):
            if ci == prev_match_indices[mi]:
                self.labels[ci] = 'S'
        
    def set_left_labels(self, label):
        self.labels = [label if l == 0 else l for l in self.labels]
        
        


class Comparator:
    
    # Compare chunks not sentence by sentence, but compare all possbile pair of subsets of sentences

    def __init__(self, model, t1, t2):
        self.model = model
        self.doc1 = Document(t1)
        self.doc2 = Document(t2)

    
    def label_same(self, thrs=0.99):
        '''
        First, match sentence at t_k to sentence at t_k+1
        Then, match consecutive sentences at t_k to consecutive sentences at t_k+1
        '''
        if len(self.doc1.sentences) == 0 or len(self.doc2.sentences) == 0:
            return
        self._label_same_sentences(thrs)
        self._label_same_paragraphs(thrs)
        

    def _label_same_sentences(self, thrs=0.99):
        '''
        Match sentence in t_k to sentence in t_k+1
        '''
        similarity = self.model.similarity(self.doc1.sentences, self.doc2.sentences).round(2)
        self.doc2.set_match_indices(similarity, thrs)
        self.doc1.set_match_indices(similarity.transpose(1, 0), thrs)
        self.doc2.set_same_labels(self.doc1.match_indices)
        self.doc1.set_same_labels(self.doc2.match_indices)


    def _label_same_paragraphs(self, thrs=0.99, max_len=5):
        '''
        For left chunks after label_same_sentence, 
        determine whether there are any consecutive sentences in the chunk are matched
        '''
        new, old = self.get_unlabeled_chunks()
        for (os, oe), (ns, ne) in zip(old, new):
            if os >= oe:               
                continue
            
            # find all consecutive indices for chunks (length < max_len)
            oe += 1
            ne += 1
            indices_old = [[os - 1, o] for o in range(os, min(oe, os + max_len))]\
                        + [[o1, o2] for o1 in range(os, oe) for o2 in range(o1 + 1, oe + 1) if o2 - o1 <= max_len]
            indices_new = [[ns - 1, n] for n in range(ns, min(ne, ne + max_len))]\
                        + [[n1, n2] for n1 in range(ns, ne) for n2 in range(n1 + 1, ne + 1) if n2 - n1 <= max_len]
                        
            # find all consecutive sentences for chunks
            sentences_old = [' '.join(self.doc1.sentences[o1:o2]) for o1, o2 in indices_old]
            sentences_new = [' '.join(self.doc2.sentences[n1:n2]) for n1, n2 in indices_new]
            
            # exclude the first and the last
            indices_old = indices_old[1:-1]
            indices_new = indices_new[1:-1]
            sentences_old = sentences_old[1:-1]
            sentences_new = sentences_new[1:-1]
            
            # calculate similarity of all possible sub-chunks
            sim = self.model.similarity(sentences_old, sentences_new).round(2)

            # select indices where the similarity is greater than thrs
            indices = [(r, c) for r, c in zip(*np.where(sim >= thrs))]
            indices.sort(reverse=True, key=lambda x: sim[x[0], x[1]])

            # update labels & match indices of doc2
            for r, c in indices:
                # print(indices_old[r], indices_new[c], len(self.doc1.labels), len(self.doc2.labels))
                for i in range(*indices_old[r]):
                    if i >= len(self.doc1.labels):
                        break
                    self.doc1.labels[i] = 'S'
                    if self.doc1.match_indices[i] == -1:
                        self.doc1.match_indices[i] = [tuple(indices_new[c])]
                    elif type(self.doc1.match_indices[i]) == list:
                        self.doc1.match_indices[i].append(tuple(indices_new[c]))
                        
                for i in range(*indices_new[c]):
                    if i >= len(self.doc2.labels):
                        break
                    self.doc2.labels[i] = 'S'
                    if self.doc2.match_indices[i] == -1:
                        self.doc2.match_indices[i] = [tuple(indices_old[r])]
                    elif type(self.doc2.match_indices[i]) == list:
                        self.doc2.match_indices[i].append(tuple(indices_old[r]))
    
    
    def get_label_same_result(self):
        '''
        After label same, get results of label_same
        '''
        # get same
        s_old = self.doc1.get_sentences_from_label('S', return_chunks=True)
        s_new = self.doc2.get_sentences_from_label('S', return_chunks=True)
        
        # get unlabeled chunks
        new, old = self.get_unlabeled_chunks()
        
        # label NS (not same)
        ns_old = {'indices':[], 'sentences':[]}
        ns_new = {'indices':[], 'sentences':[]}
        n_new = {'indices':[], 'sentences':[]}
        
        for o, n in zip(old, new):
            # new
            if o[0] == o[1] or o[0] == len(self.doc1.sentences):
                indices_new = list(range(n[0], n[1]))
                n_new['indices'].append(indices_new)
                n_new['sentences'].append([self.doc2.sentences[i] for i in indices_new])
                for i in indices_new:
                    self.doc2.labels[i] = 'N'
            # not same
            else:
                indices_old = list(range(o[0], min(o[1], len(self.doc1.labels))))
                indices_new = list(range(n[0], n[1]))
                ns_old['indices'].append(indices_old)
                ns_new['indices'].append(indices_new)
                ns_old['sentences'].append([self.doc1.sentences[i] for i in indices_old])
                ns_new['sentences'].append([self.doc2.sentences[i] for i in indices_new])
                for i in indices_old:
                    self.doc1.labels[i] = 'NS'
                for i in indices_new:
                    self.doc2.labels[i] = 'NS'
        
        deleted = self.doc1.get_sentences_from_label(0)
        d_old = {'indices':list(deleted.keys()),
                 'sentences':list(deleted.values())}
        
        result_old = {'S': s_old, 'NS': ns_old, 'D': d_old}
        result_new = {'S': s_new, 'NS': ns_new, 'N': n_new}
                
        return result_old, result_new
    
    
    def get_unlabeled_chunks(self):
        '''
           Returns : chunks (of new article) & matched_chunks (of old article)
           Chunk : a list of [start_index, end_index + 1]
           Thus, you can use the chunk for index slicing directly
        '''
        left_dic = self.doc2.get_sentences_from_label(0)
        chunks = []
        match_chunks = []
        
        for idx, sen in left_dic.items():
            
            # find start index of matched chunks
            if idx == 0:
                start_idx = 0
            else:
                start_idx = self.doc2.match_indices[idx - 1] 
                # match sentence
                if type(start_idx) == int:
                    start_idx = start_idx + 1 if start_idx != -1 else -1
                # match chunk
                else:
                    start_idx = max([i[1] for i in start_idx])

            # break if doc1 ended
            if start_idx >= len(self.doc1.match_indices):
                chunks.append([idx, len(self.doc2.match_indices)])
                match_chunks.append([len(self.doc1.match_indices), len(self.doc1.match_indices)])
                break

            # if start point, add new chunk
            if start_idx != -1:
                chunks.append([idx, idx+1])
                match_chunks.append([start_idx, -1])

            # find end index of matched chunks
            end_idx = self.doc2.match_indices[idx+1] \
                    if idx + 1 < len(self.doc2.match_indices) else len(self.doc2.match_indices) - 1

            # match chunk 
            if type(end_idx) == list:
                end_idx = min([i[0] for i in end_idx])

            # if end point, update end indices of the last chunk
            if end_idx != -1:
                chunks[-1][-1] = idx + 1
                match_chunks[-1][-1] = end_idx 
                
        # check the last end index of match_chunks
        if len(match_chunks) > 0 and match_chunks[-1][-1] == -1:
            match_chunks[-1][-1] = len(self.doc1.match_indices)
            
        return chunks, match_chunks
    
    


class Classifier:
    
    # classify new & updated sentences using NLI task
    
    def __init__(self, sim_model, clf_model, batch_size=32):
        self.sim_model = sim_model
        self.clf_model = clf_model.to('cuda')
        self.batch_size = batch_size
        self.label_dict = {'ctd': 0, 'ntr': 1, 'etl': 2}
    
    def _run_clf_model(self, batches):
        # classify batches
        res = []
        for i in range(0, len(batches), self.batch_size):
            with torch.no_grad():
                logprobs = self.clf_model.predict('mnli', batches[i : i + self.batch_size]).detach().cpu()
                res.append(logprobs.argmax(dim=1))
        res = torch.cat(res)
        return res
    
    def _run_batches(self, old, new, max_len=512):
        # get batches
        batches = collate_tokens(
                    [self.clf_model.encode(o, n) for o in old for n in new], pad_idx=1
                    )
        batches = batches[:,:max_len].to('cuda')
        batch_results = self._run_clf_model(batches)
        return batch_results
    
    def _get_label_result(self, result, label):
        label = self.label_dict[label]
        indices = torch.where(result == label)[0]
        return indices
    
    def _classify_ctd(self, sim, res, old, new, old_sid, new_sid, thrs=0.6):

        # pre-remove sentence where max sim >= thrs 
        old_indices = np.where(sim.max(1).round(2) >= thrs)[0]
        new_indices = np.where(sim.max(0).round(2) >= thrs)[0]

        indices = self._get_label_result(res, 'ctd')
        ctd_indices_old, ctd_indices_new = [], []
        for i in indices:
            old_idx = int(i // len(new))
            new_idx = int(i % len(new))
            if old_idx in old_indices and new_idx in new_indices:
                ctd_indices_old.append(int(old_idx))
                ctd_indices_new.append(int(new_idx))

        if len(ctd_indices_old) == 0:
            return [], []
        
        old_ = [old[i] for i in set(ctd_indices_old)]
        new_ = [new[i] for i in set(ctd_indices_new)]
        
        res = self._run_batches(new_, old_)
        indices = self._get_label_result(res, 'ctd')
        ctd_indices = []
        ctd_sentences = []
        for i in indices:
            old_idx = int(i % len(new_))
            new_idx = int(i // len(new_))
            if old_idx in ctd_indices_old and new_idx in ctd_indices_new:
                ctd_indices.append([int(new_idx + new_sid), int(old_idx + old_sid)])
                ctd_sentences.append([old[int(new_idx)], new[int(old_idx)]])

        return ctd_indices, ctd_sentences
    
    
    def _classify_nch(self, res, new, new_sid):
        indices = self._get_label_result(res, 'etl')
        nch_indices = [int(new_sid + i) % len(new) for i in indices]
        return nch_indices
    
    def _classify_new(self, sim, new, new_sid, ctd_res, nch_res, thrs=0.7):
        new_indices = np.where(sim.max(0).round(2) <= thrs)[0]
        new_indices = [new_sid + int(i) for i in new_indices]
        ctd_res_new = [i[1] for i in ctd_res]
        new_indices = [i for i in new_indices if i not in ctd_res_new and i not in nch_res]
        new_sentences = [new[i - new_sid] for i in new_indices] 
        return new_indices, new_sentences
    
            
    def _run_labeling(self, old, new, old_sid, new_sid):
        with torch.no_grad():
            sim = self.sim_model.similarity(old, new)
        res = self._run_batches(old, new)
        ctd_res, ctd_sentences = self._classify_ctd(sim, res, old, new, old_sid, new_sid)
        nch_res = self._classify_nch(res, new, new_sid)
        new_res, new_sentences = self._classify_new(sim, new, new_sid, ctd_res, nch_res)        
        return ctd_res, ctd_sentences, new_res, new_sentences
        

    def label_changed_and_new_sentences(self, old, new):
        # label changed and new
        result_old = {'C': {'indices': [], 'sentences':[]}}
        result_new = {'N': {'indices': [], 'sentences':[]}}
        
        for ind_old, ind_new, sen_old, sen_new in zip(old['NS']['indices'], 
                                                      new['NS']['indices'], 
                                                      old['NS']['sentences'], 
                                                      new['NS']['sentences']):
            if len(ind_old) == 0:
                continue
            ctd_res, ctd_sentences, new_res, new_sentences = self._run_labeling(sen_old, 
                                                                                sen_new, 
                                                                                ind_old[0], 
                                                                                ind_new[0])
            result_old['C']['indices'].extend(ctd_res)
            result_new['N']['indices'].extend(new_res)
            result_old['C']['sentences'].extend(ctd_sentences)
            result_new['N']['sentences'].extend(new_sentences)
            
        # update same
        if 'S' in old and 'indices' in old['S']:
            old['S'] = list_to_dict(old['S']) 

        # update changed
        old['C'] = result_old['C']

        # update new
        if 'N' in new and 'indices' in new['N']:
            new['N']['indices']= [i for lst in new['N']['indices'] for i in lst]
            new['N']['indices'].extend(result_new['N']['indices'])
            new['N']['sentences'].extend(result_new['N']['sentences'])
        else:
            new['N'] = result_new['N']
        
        # remove unnecessary results
        del old['NS'], new['S']     
        
        return old, new
    
class Filter:
    
    def __init__(self, model):
        with torch.no_grad():
            self.model = model
            
    def _run_prompt(self, messages, temp=0):
        response = openai.ChatCompletion.create(
            model = 'gpt-4-1106-preview',
            messages = messages,
            temperature = temp,
            max_tokens = 256
        )
        response = response['choices'][0]['message']['content']
        return response

    def _get_prompt(self, src, trg):
        return 'Please examine the target sentence(s) and determine whether each sentence of the target sentence(s) \"contradict\" the source sentence(s). '+\
                f'Target sentence(s): {trg} \n\n'+\
                f'Source sentence(s): {src} \n\n'+\
                'Are there target sentence(s) that \"contradict\" to source sentence(s)? If no, answer only "No". '+\
                f'Otherwise, answer only the list of pairs of indices of contradictory sentences. The index starts from 0.\n'+\
                f'The format is as follows: \n'+\
                '[(index of source sentence, index of contradictory target sentence), ... ]'
                    
    def _result_to_list(self, res):
        if 'no' in res.lower()[:5]:
            return []
        elif '[' in res and ']' in res:
            res = res[res.index('[') : res.rindex(']') + 1]
            res = res.replace("\'", "'").replace("'", "\'").replace('\n', ' ')
            try: 
                return eval(res)
            except Exception as e:
                return []
        else:
            return []
    
    def _sim_filter(self, old, new, thrs=0.6):       
        
        # pre-remove sentence where max sim <= thrs & new not in N_res
        sim = self.model.similarity(old, new)
        old_indices = np.where(sim.max(1).round(2) >= thrs)[0]
        new_indices = np.where(sim.max(0).round(2) >= thrs)[0]
        old_indices = [int(i) for i in old_indices]
        new_indices = [int(i) for i in new_indices]
        old_sent = [old[i] for i in old_indices]
        new_sent = [new[i] for i in new_indices]
        filt_indices = [(i, j) for i, j in zip(old_indices, new_indices)]
        filt_sent = [(i, j) for i, j in zip(old_sent, new_sent)]
        
        return filt_indices, filt_sent
    
    def run_filter(self, ctd): # {'indices': [[old:new]..], 'sentences':[[old_sen, new_sen]..]}
        
        # get sentences
        ctd_old = {}
        ctd_new = {}
        for (i_o, i_n), (so, sn) in zip(ctd['indices'], ctd['sentences']):
            ctd_old[i_o] = so
            ctd_new[i_n] = sn
        ctd_idx_old = list(ctd_old.keys())
        ctd_idx_new = list(ctd_new.keys())
        ctd_sen_old = list(ctd_old.values())
        ctd_sen_new = list(ctd_new.values())
        
        # api call
        messages = [{"role" : "system", "content" : "You are a helpful assistant."}]
        prompt = self._get_prompt(ctd_sen_old, ctd_sen_new)
        messages.extend([{"role" : "user", "content": prompt}])
        res = self._run_prompt(messages)
        
        
        # post process
        res = self._result_to_list(res)
        if len(res) == 0:
            return {'indices': [], 'sentences': []}
        
        # get result
        ctd_idx_res = []
        for i, j in res:
            if i < len(ctd_idx_old) and j < len(ctd_idx_new):
                ctd_idx_res.append((i, j))
        ctd_sen_old = [ctd_sen_old[i[0]] for i in ctd_idx_res]
        ctd_sen_new = [ctd_sen_new[i[1]] for i in ctd_idx_res]
        
        # sim check
        ctd_idx_res, ctd_sen_res = self._sim_filter(ctd_sen_old, ctd_sen_new)
        ctd_idx_res = [(ctd_idx_old[i], ctd_idx_new[j]) for i, j in ctd_idx_res]
        print(ctd_idx_res)
        print(ctd_sen_res)
        return {'indices': ctd_idx_res, 'sentences': ctd_sen_res}

    
    
if __name__ == "__main__":
    
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    
    p1 = torch.load('trash_1027/covid-19_10.pt')
    p2 = torch.load('trash_1027/covid-19_11.pt')
    
    t8 = split_article_into_sentences(p1)
    t9 = split_article_into_sentences(p2)
    sim = model.similarity(t8, t9)
    
    comp = Comparator(10, 11, p1, p2)
    comp.label_same_sentences(sim)
    
    