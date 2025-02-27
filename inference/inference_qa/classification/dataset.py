from torch.utils.data import Dataset
import json
import torch
import random
from glob import glob
from string import Template


class FilterDataset(Dataset):
    
    def __init__(
        self,
        dataset: dict = None,
        data_path: str = None,
        num_train_data: int = 200,
        num_val_data: int = 4,
        mode: str = 'train',
        top_k: int = 3,
        concat: bool = False,
        wo_context: bool = False,
        retrieval_key: str = 'retrieval'
    ):

        self.top_k = top_k
        self.mode = mode
        self.retrieval_key = retrieval_key
        self.wo_context = wo_context
        
        # train or val
        if data_path and mode in ['train', 'val']:
            # read data
            if 'jsonl' not in data_path:
                with open(data_path) as f:
                    datas = json.load(f)
                    datas = list(datas.items())
            else:
                with open(data_path) as f:
                    datas = [json.loads(i) for i in f.readlines()]
                    datas = [[i['id'], i] for i in datas]
            
            if mode == 'train':
                self.num_data = num_train_data
                datas, output = self.select_data(datas[:30000])
                self.datas = datas
                self.labels = output
    
            elif mode == 'val':
                self.num_data = num_val_data
                datas, output = self.select_data(datas[30000:])
                self.datas = datas
                self.labels = output

            print(len(self.datas), len(self.labels))

        elif dataset:
            if not concat:
                self.datas = self.partition_data(dataset)
            else:
                self.datas = self.concat_data(dataset)
            self.labels = None

    def __getitem__(self, idx):
        datapoint = self.datas[idx]
        qaid = datapoint[0]
        prompt = datapoint[1]['prompt']
        meta = datapoint[1]
        if self.labels is not None:
            label = self.labels[idx]
            return qaid, prompt, meta, label
        else:
            return qaid, prompt, meta
    
    def __len__(self):
        return len(self.datas)
    
    
    def collate_fn(self, batch):
        qaids = [i[0] for i in batch]
        prompts = [i[1] for i in batch]
        metas = [i[2] for i in batch]
        if self.labels is not None:
            labels = [i[3] for i in batch]
            return qaids, prompts, metas, torch.tensor(labels)
        else:
            return qaids, prompts, metas
    
    
    def get_prompt(self, inputs):
        if len(inputs['context']) > 0:
            template = Template("Use the following context to answer the question.\nContext: $context\nQuestion: $question\nAnswer:") # Use the following pieces of context
            return template.substitute(inputs)
        else:
            template = Template("Question: $question\nAnswer:") # Use the following pieces of context
            return template.substitute(inputs)
    
    
    def select_data(self, datas):
        new_data = {'CHANGED': [], 'NEW': [], 'SAME': []}
        new_label = {'CHANGED': [], 'NEW': [], 'SAME': []}
        for qid, qitem in datas:
            qtype = qitem['type']
            if len(new_data[qtype]) >= self.num_data:
                continue
            if qtype == 'SAME':
                find_hit = 1
                label = [1., 0., 0.]
            elif qtype == 'CHANGED':
                find_hit = 0
                label = [0., 1., 0.]
            else:
                find_hit = 0
                label = [0., 0., 1.]
            hit_idx = -1
            retrievals = list(qitem[self.retrieval_key])
            del qitem[self.retrieval_key]
            for i, ret in enumerate(retrievals[:self.top_k]):
                if ret['hit'] == find_hit:
                    qitem['context'] = ret['document']
                    qitem[self.retrieval_key] = ret
                    qitem['prompt'] = self.get_prompt({'question': qitem['question'], 
                                                       'context': qitem['context']})
                    new_data[qtype].append([qid, dict(qitem)])
                    new_label[qtype].append(label) # enough
                    hit_idx = i
                    break
                     
        new_data = [i for m in new_data.values() for i in m]
        new_label = [i for m in new_label.values() for i in m]
        indices = list(range(len(new_data)))
        random.shuffle(indices)
        new_data = [new_data[i] for i in indices]
        new_label = [new_label[i] for i in indices]
        
        return new_data, new_label
    
    
    def partition_data(self, datas):
        new_data = []
        for qid, qitem in datas:
            retrievals = list(qitem[self.retrieval_key])    
            sub_retrievals = retrievals[:self.top_k]
            if 'document' not in sub_retrievals[0]:
                sub_retrievals = [i['output'] for i in sub_retrievals]
            for ret in sub_retrievals:
                qitem['context'] = ret['document']
                qitem[self.retrieval_key] = ret
                qitem['prompt'] = self.get_prompt({'question': qitem['question'], 
                                                   'context': qitem['context']})
                new_data.append([qid, dict(qitem)])
            
        return new_data
    
    def concat_data(self, datas):
        new_data = []
        for qid, qitem in datas:
            retrievals = list(qitem[self.retrieval_key])    
            sub_retrievals = retrievals[:self.top_k]
            context = ''
            if not self.wo_context:
                for ret in sub_retrievals:
                    context += ret['document']
            qitem['context'] = context
            qitem['prompt'] = self.get_prompt({'question': qitem['question'], 
                                                'context': qitem['context']})
            new_data.append([qid, dict(qitem)])
            
        return new_data
