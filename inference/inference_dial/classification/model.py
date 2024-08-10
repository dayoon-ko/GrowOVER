from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from langchain.llms import HuggingFacePipeline
import torch
import torch.nn as nn 
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
import torch.distributed as dist
from accelerate import Accelerator
from typing import Optional, List 
from transformers import LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import Logger
import json


def load_model_state_dict(model: torch.nn.Module, path: str) -> None:
    if dist.get_rank() == 0:
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False) #, assign=True)  # assign checkpoint tensor
        del state_dict
    
def get_model(
        accelerator: Accelerator,
        llama_config_dir: str = '/gallery_louvre/dayoon.ko/research/llama2/checkpoints',
        llama_ckpt_path: str = None,
        train_pred: bool = False,
        pred_ckpt_path: str = None,
        logger: Logger = None, 
    ):
    if logger:
        logger.info('Initialize model...')
    model = LlamaForCausalLM.from_pretrained(
        llama_config_dir,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        return_dict=True
    )
    if logger:
        logger.info('Model initialized...')
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = LlamaTokenizer.from_pretrained(llama_config_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    if llama_ckpt_path is not None: 
        if logger:
            logger.info(f'Load checkpoint from {llama_ckpt_path}...')
        load_model_state_dict(model, llama_ckpt_path)
        if logger:
            logger.info('Checkpoint loaded...')
    
    if train_pred:
        model = LlamaForFilter(accelerator, 
                               model, 
                               tokenizer, 
                               logger=logger
                               )
    else:
        model = LlamaForFilter(accelerator, 
                               model, 
                               tokenizer, 
                               train=False, 
                               ckpt_path=pred_ckpt_path,
                               logger=logger
                               )
        
    return tokenizer, model


class LlamaForFilter:
    
    def __init__(self, 
                 accelerator: Accelerator,
                 model: LlamaForCausalLM,
                 tokenizer: LlamaTokenizer,
                 hidden_size: int = 4096,
                 train: bool = True,
                 ckpt_path: str = None,
                 logger: Logger = None
                 ):
        
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side='left'
        self.hidden_size = hidden_size
        if logger:
            self.logger = logger
        
        pred_head = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.SiLU(),
        )
        pred_head.cuda()
        self.pred_head = accelerator.prepare(pred_head)
        
        if ckpt_path:
            model = accelerator.load_state(ckpt_path)
        
        if train:
            self.loss_fct = CrossEntropyLoss()
        
    def __call__(self, 
                 inputs, 
                 labels, 
                 ):
        self.train()
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
        
        # llama inference
        max_length = tokenized_inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**tokenized_inputs, 
                                                output_scores=True,
                                                output_hidden_states=True,
                                                max_new_tokens=100, 
                                                return_dict_in_generate=True
                                                )
        
        hidden_states = outputs.hidden_states[0][-1][:, -1, :].squeeze().type(torch.float32) # #token, #layer, #batch, #vocab, hidden_size
        del outputs
        
        # forward pass of training
        predictions = self.pred_head(hidden_states).squeeze()
        loss = self.loss_fct(predictions, labels.cuda())
            
        return loss
    
    
    def inference(self, 
                  inputs, 
                  labels=None
                  ):
        self.eval()
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
        
        # llama inference
        max_length = tokenized_inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**tokenized_inputs, 
                                                output_scores=True,
                                                output_hidden_states=True,
                                                max_new_tokens=100, 
                                                return_dict_in_generate=True
                                                )

        sequences = outputs.sequences.detach().cpu()[:, max_length:] # batch, max_new_token
        hidden_states = outputs.hidden_states[0][-1][:, -1, :].squeeze().type(torch.float32) # #token, #layer, #batch, #vocab, hidden_size
        del outputs
        
        texts = [] 
        # generate text 
        for i, seq in enumerate(sequences):
            sid, eid = self.find_indices(seq)
            texts.append(self.tokenizer.decode(seq[sid:eid]))
            #texts.append(self.tokenizer.decode(seq))
        
        # forward pass of training
        predictions = self.pred_head(hidden_states).squeeze().cpu().detach() 
        pred_probs = nn.Softmax(-1)(predictions)
        pred_indices = pred_probs.argmax(1)
        pred_labels = torch.zeros(pred_probs.shape).scatter(1, pred_indices.unsqueeze(1), 1.0)
        
        if labels is None:
            return [{'prob': p, 'label': i, 'text': t} for p, i, t in zip(pred_probs[:,0].squeeze().tolist(), pred_indices.tolist(), texts)]    
            
        # for validation
        else:
            # cal loss
            labels = labels.detach().cpu()
            accuracy = torch.sum(pred_indices == labels.argmax(1)) / predictions.shape[0]
            loss = self.loss_fct(predictions, labels)
            if self.logger:
                #self.logger.info(str(nn.Softmax(-1)(predictions)))
                self.logger.info('predicted labels\n' + str(pred_labels))
                self.logger.info('labels\n' + str(labels))
                self.logger.info('accuracy: ' + str(accuracy))
                self.logger.info(json.dumps(texts, indent=2))
            
            return loss, accuracy, texts 
        
    
    def get_params(self):
        return self.pred_head.parameters()
    
    def eval(self):
        self.mode = 'eval'
        self.pred_head.eval()
        
    def train(self):
        self.mode = 'train'
        self.pred_head.train()
        
    def find_indices(self, seq):
        # find start
        start_idx = 0
        end_idx = seq.shape[0]
        newline_idx = (seq == 13).nonzero().squeeze(-1)
        
        # no newline
        if len(newline_idx) == 0:
            return start_idx, end_idx
        
        # set start_idx
        while start_idx in newline_idx:
            start_idx += 1
            
        # set end_idx
        try:
            end_idx = newline_idx[(newline_idx > start_idx).nonzero().squeeze(-1).tolist()[0]]
        except:
            end_idx = seq.shape[0]
        # return
        return start_idx, end_idx