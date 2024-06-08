import torch
import numpy as np
import os
import openai
from tqdm import tqdm
from simcse import SimCSE
from nltk import sent_tokenize
import re
import os


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
        if len(p) > 1:
            for s in p:
                sentences.append(s)
    return sentences
    

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


    def label_same_sentences(self, thrs=0.99):
        '''
        Match sentence in t_k to sentence in t_k+1
        '''
        similarity = self.model.similarity(self.doc1.sentences, self.doc2.sentences).round(2)
        self.doc2.set_match_indices(similarity, thrs)
        self.doc1.set_match_indices(similarity.transpose(1, 0), thrs)
        self.doc2.set_same_labels(self.doc1.match_indices)
        self.doc1.set_same_labels(self.doc2.match_indices)


    def label_same_paragraphs(self, thrs=0.99, max_len=5):
        '''
        For left chunks after label_same_sentence, 
        determine whether there are any consecutive sentences in the chunk are matched
        '''
        new, old = self.get_unlabeled_chunks()
        for (os, oe), (ns, ne) in zip(old, new):
            if os >= oe:
                #if os > oe: print(os, oe, ns, ne)                    
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
                    


    def label_same(self, thrs=0.99):
        '''
        First, match sentence at t_k to sentence at t_k+1
        Then, match consecutive sentences at t_k to consecutive sentences at t_k+1
        '''
        if len(self.doc1.sentences) == 0 or len(self.doc2.sentences) == 0:
            return
        self.label_same_sentences(thrs)
        self.label_same_paragraphs(thrs)
        
        
    
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
    
    


class Prompter:
    
    # The folling process is executed by label_changed_and_new_sentences method
    # 1. Generate C, D, N prompts and run api calls 
    # 2. Get results from the answers
    # 3. Find same one more time
    
    def __init__(self, model):
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
    
    
    def _get_prompt(self, src, trg, prompt_type):
        # prompt_type should be either 'contradict' or 'new'
        
        if prompt_type == 'contradict':
            return 'Please examine the target sentence(s) and determine whether each sentence of the target sentence(s) contradicts the source sentence(s).  \n'+\
                    f'Target sentence(s): {trg} \n'+\
                    f'Source sentence(s): {src} \n'+\
                    'Are there target sentence(s) that contradict to source sentence(s)? If no, answer only "No".'+\
                    f'Otherwise, answer only the list of pairs of indices of contradictory sentences. The index starts from 0.\n'+\
                    f'The format is as follow: \n'+\
                    '[(index of source sentence, index of contradictory target sentence), ... ]'
                    
        
        if prompt_type == 'new':
            return 'Please examine the target sentence(s) and determine whether each sentence of the target sentence(s) introduces additional information compared to source sentence(s). \n'+\
                    f'Target sentence(s): {trg} \n'+\
                    f'Source sentece(s): {src} \n'+\
                    'Are there target sentence(s) that provide additional information? If no, answer only "No".'+\
                    f'Otherwise, answer only the list of indices of target sentence(s) containing additional information. The index starts from 0.\n'+\
                    f'The format is as follow: \n'+\
                    '[ index of target sentence with additional information, ... ]'
                    
        
    def _postprocess_result(self, res):
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
        
                
    def _get_label_C_result(self, prompt, old, new, old_sid, new_sid, thrs=0.6):
        
        # pre-remove sentence where max sim <= thrs
        sim = self.model.similarity(old, new)
        old_indices = np.where(sim.max(1).round(2) >= thrs)[0]
        new_indices = np.where(sim.max(0).round(2) >= thrs)[0]
        old_, new_ = [old[i] for i in old_indices], [new[i] for i in new_indices]
        
        # run prompt
        prompt.extend([{"role" : "user", "content": self._get_prompt(new_, old_, prompt_type='contradict')}])
        res = self._run_prompt(prompt)
        
        # post process the result
        res = self._postprocess_result(res) # [(0,0), (1,3)]
        if len(res) == 0:
            return {}
        
        # index synchronization
        res = [(new_indices[i], old_indices[j]) for i, j in res]
        print('res:', res)
        
        # verify result using sentence similarity
        new_, old_ = [], []
        for i, j in res:
            if i < len(new) and j < len(old):
                new_.append(new[i])
                old_.append(old[j])
                
        if len(old_) == 0:
            return {}
            
        sim = self.model.similarity(old_, new_).diagonal()
        is_valid = sim > thrs # [False, True, False]
        
        # get result
        if np.all(is_valid == False):
            return {}
        else:
            return {old_sid + int(old_idx) : new_sid + int(new_idx) 
                    for i, (new_idx, old_idx) in enumerate(res) if is_valid[i]}
        
        
    def _get_label_N_result(self, prompt, old, new, old_sid, new_sid, C_res, thrs=0.7):
        
        # pre-remove sentence where max sim >= thrs
        sim = self.model.similarity(old, new)
        old_indices = np.where(sim.max(1).round(2) <= thrs)[0]
        new_indices = np.where(sim.max(0).round(2) <= thrs)[0]
        old_, new_ = [old[i] for i in old_indices], [new[i] for i in new_indices]
        
        # run prompt
        prompt.extend([{"role" : "user", "content": self._get_prompt(new_, old_, prompt_type='new')}])
        res = self._run_prompt(prompt)
        
        # postprocess
        res = self._postprocess_result(res) # [0, 1, 3]
        res = [i for i in res if i < len(old)]
        if len(res) == 0:
            return {}
        
        # index synchronization
        res = [int(old_indices[i]) for i in res]
        
        # verify result using sentence similarity
        old = [old[i] for i in res if i < len(old)]
        sim = self.model.similarity(old, new)
        argmax_indices = list(sim.argmax(1))
        is_valid = sim.max(1) > thrs # [False, True, False]
        
        # get result
        new_res = {}
        for i in range(len(res)):
            if i >= len(old):
                break
            k = old_sid + res[i]
            s = new_sid + int(argmax_indices[i]) if is_valid[i] else None
            if k not in C_res:
                new_res[k] = s
        return new_res
    
    
    def _run_label_prompt(self, old, new, old_sid, new_sid):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # identify difference
        prompt = [{"role" : "system", "content" : "You are a helpful assistant."}]
        
        # find C
        C_res = self._get_label_C_result(prompt, old, new, old_sid, new_sid)
        print('ctd:', C_res)
        
        # find D
        # D_res = self._get_label_N_result(prompt, old, new, old_sid, new_sid, C_res.keys())
        
        # find N
        N_res = self._get_label_N_result(prompt, new, old, new_sid, old_sid, C_res.values())
        print('new:', N_res)
        
        return C_res, N_res
    
    
    def _run_label_last_same(self, old, new, old_sid, new_sid, C_res, N_res, thrs=0.85):
        
        # select sentences not in C, D, N
        old_indices, new_indices = [], []
        for i, j in zip(range(old_sid, len(old) + old_sid), range(new_sid, len(new) + new_sid)):
            if i not in C_res.keys() and j not in C_res.values():
                old_indices.append(i)
                new_indices.append(j)
                    
        # verify same using sentence similarity
        old_ = [old[i - old_sid] for i in old_indices]
        new_ = [new[i - new_sid] for i in new_indices]
        if len(old_) == 0 or new_ == 0:
            return {}
        sim = self.model.similarity(old_, new_)
        argmax_indices = list(sim.argmax(1))
        is_valid = sim.max(1) > thrs # [False, True, False]
        
        # get result
        return {old_sid + i : new_sid + int(argmax_indices[i]) 
                for i in range(len(old_)) if is_valid[i]}
        
            
    def label_changed_and_new_sentences(self, old, new):
        result_old = {'C': {}, 'S2': {},}
        result_new = {'N': {}}
        
        for ind_old, ind_new, sen_old, sen_new in zip(old['indices'], new['indices'], old['sentences'], new['sentences']):
            
            if len(ind_old) == 0:
                continue
            
            # prompt GPT
            C_res, N_res = self._run_label_prompt(sen_old, sen_new, ind_old[0], ind_new[0])

            result_old['C'].update(C_res)
            result_new['N'].update(N_res)
            
            # last same
            S_res = self._run_label_last_same(sen_old, sen_new, ind_old[0], ind_new[0], C_res, N_res)
            result_old['S2'].update(S_res)
            print('same:', S_res)
            
        return result_old, result_new
    
    
if __name__ == "__main__":
    
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    
    p1 = torch.load('trash_1027/covid-19_10.pt')
    p2 = torch.load('trash_1027/covid-19_11.pt')
    
    t8 = split_article_into_sentences(p1)
    t9 = split_article_into_sentences(p2)
    sim = model.similarity(t8, t9)
    
    comp = Comparator(10, 11, p1, p2)
    comp.label_same_sentences(sim)
    comp.label_changed_and_new_sentences()
    comp.label_del_sentences()