
    
from torch.utils.data import Dataset
import json
import torch
import random
from glob import glob
from string import Template


class FilterDataset(Dataset):
    
    def __init__(
        self,
        month: int = 9,
        dataset: dict = None,
        data_path: str = '../datasets/09/vectorstore/retrievalturn.jsonl',
        num_train_data: int = 512,
        num_val_data: int = 128,
        mode: str = 'train',
        top_k: int = 3,
        concat: bool = False,
        wo_context: bool = False,
        query: str = 'retrieval',
    ):

        self.top_k = top_k
        self.mode = mode
        self.query = query
        self.wo_context = wo_context
        
        # train or val
        if data_path and mode in ['train', 'val']:
            
            # read data
            with open(data_path) as f:
                datas = [list(list(json.loads(i).items())[0]) for i in f.readlines()]
                datas = sorted(datas, key=lambda x: int(x[0]))
            
            if mode == 'train':
                self.num_data = num_train_data
                datas, output = self.select_data(datas[:112500])
                self.datas = datas
                self.labels = output
    
            elif mode == 'val':
                self.num_data = num_val_data
                datas, output = self.select_data(datas[112500:])
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
    
    
    def get_hist_prompt(self, inputs):
        template = Template("Use the following context to answer the question.\nContext: $context\nChat history: \n$history\nQuestion: $user\nAnswer:") 
        return template.substitute(inputs)


    def get_non_hist_prompt(self, inputs):
        template = Template("Use the following context to answer the question.\nContext: $context\nQuestion: $user\nAnswer:")
        return template.substitute(inputs)
    
    
    def select_data(self, datas):
        new_data = {'CONTRADICT': [], 'NEW': [], 'SAME': []}
        new_label = {'CONTRADICT': [], 'NEW': [], 'SAME': []}
        for tid, turn in datas:
            qtype = turn['sentence_type']
            if len(new_data[qtype]) >= self.num_data:
                continue
            if qtype == 'SAME':
                find_hit = 1
                label = [1., 0., 0.]
            elif qtype == 'CONTRADICT':
                find_hit = 0
                label = [0., 1., 0.]
            else:
                find_hit = 0
                label = [0., 0., 1.]
            hit_idx = -1
            retrievals = list(turn['retrieval'])
            for i, ret in enumerate(retrievals[:self.top_k]):
                if ret['hit'] == find_hit:
                    turn['context'] = ret['document']
                    turn['retrieval'] = ret
                    if len(turn['history']) > 0:
                        turn['prompt'] = self.get_hist_prompt(turn)               
                    else:
                        turn['prompt'] = self.get_non_hist_prompt(turn)
                    new_data[qtype].append([tid, dict(turn)])
                    new_label[qtype].append(label) # enough
                    hit_idx = i
                    break
        print('CONTRADICT:', len(new_data['CONTRADICT']))
        print('NEW:', len(new_data['NEW']))
        print('SAME:', len(new_data['SAME']))
        new_data = [i for m in new_data.values() for i in m]
        new_data = dict(new_data)
        if self.mode == 'train':
            with open('train.json', 'w') as f:
                json.dump(new_data, f, indent=2)
            exit()
        new_label = [i for m in new_label.values() for i in m]
        indices = list(range(len(new_data)))
        random.shuffle(indices)
        new_data = [new_data[i] for i in indices]
        new_label = [new_label[i] for i in indices]
        
        return new_data, new_label
    
    
    def partition_data(self, datas):
        new_data = []
        for tid, turn in datas:
            retrievals = list(turn[self.query])  
            sub_retrievals = retrievals[:self.top_k]
            for ret in sub_retrievals:
                turn['context'] = ret['document']
                turn[self.query] = ret
                if len(turn['history']) > 0:
                    turn['prompt'] = self.get_hist_prompt(turn)               
                else :
                    turn['prompt'] = self.get_non_hist_prompt(turn)
                new_data.append([tid, dict(turn)])
            
        return new_data

    def concat_data(self, datas):
        new_data = []
        for tid, turn in datas:
            retrievals = list(turn[self.query])    
            sub_retrievals = retrievals[:self.top_k]
            context = ''
            context = ''
            if not self.wo_context:
                for ret in sub_retrievals:
                    context += ret['document']
            turn['context'] = context
            if len(turn['history']) > 0:
                turn['prompt'] = self.get_hist_prompt(turn)               
            else :
                turn['prompt'] = self.get_non_hist_prompt(turn)
            new_data.append([tid, dict(turn)])
            
        return new_data