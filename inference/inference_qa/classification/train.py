from model import get_model 
from dataset import FilterDataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
import wandb
import fire
from tqdm import tqdm
import json
from utils import Logger, print_0
from datetime import timedelta
import warnings
from torch import optim
import os

warnings.filterwarnings("ignore")

import torch 
import random

torch.manual_seed(42)
random.seed(42)
print('Set random seed!')
            
def train_filter(
        month: int = 9,
        llm_config_dir: str = 'meta-llama/Llama-3.1-8B',
        ckpt_path: str = None,
        data_path: str = '../QA_dataset/09/retrievalqa.jsonl',
        save_root: str = 'results',
        batch_size: int = 16,
        lr: float = 0.00001,
        weight_decay: float = 1e-7,
        num_epochs: int = 100,
        eval_interval: int = 1,
        save_ckpt: bool = True,
        save_ckpt_path: str = '',
        num_train: int = 512,
        num_val: int = 128,
        wandb: bool = True,
        eta_min: bool = 1e-8,
    ):
    
    save_ckpt_dir = f'checkpoints/lr_{lr}_wd_{weight_decay}_{eta_min}'
    if os.path.exists(save_ckpt_dir):
        save_ckpt_dir = save_ckpt_dir + '_'
                
    # Init Logger
    accelerator = Accelerator(log_with="wandb")
    logger_fn = f'{save_ckpt_dir}_{str(lr)}'
    logger = Logger(logger_fn)
    logger.info(f'Training: ckpt_path: {ckpt_path}\tsave_root: {save_root}')
    logger.info(f'Learning Rate: {lr}')
    logger.info(f'Weight Decay: {weight_decay}')
    logger.info(f'Epoch: {num_epochs}')
    logger.info(f'Batch Size: {batch_size}')
    logger.info(f'Save checkpoint: checkpoints/lr_{lr}_wd_{weight_decay}')
    logger.info(f'Train: {num_train} / Val: {num_val} datapoints')
    
    # Init wandb
    if wandb:
        wandb_name = f'cls_entropy_lr_{lr}_wd_{weight_decay}_{num_train}'
        accelerator.init_trackers(
        project_name='cert', 
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size
            },
        init_kwargs={"wandb": {"entity": "dayoon1216",
                               "name": wandb_name}}
        )
        logger.info(f'Wandb Name: {wandb_name}')
        
    # Load models and create optimizer and scheduler
    tokenizer, model = get_model(accelerator, llm_config_dir, ckpt_path, logger=logger)
    optimizer = optim.AdamW(model.get_params(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=eta_min)

    # Get dataset
    train_dataset = FilterDataset(data_path=data_path, mode='train', num_train_data=num_train, num_val_data=num_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
    logger.info('Load train loader')
    
    val_dataset = FilterDataset(data_path=data_path, mode='val', num_train_data=num_train, num_val_data=num_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)
    logger.info('Load validation loader')
    
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )
    
    # train
    prev_val_loss = 100
    for epoch in range(num_epochs):
        results = []
        total_train_loss = 0
        for _, batch in enumerate(train_dataloader):
            model.train()
            qaids, questions, metas, labels = batch
            optimizer.zero_grad()
            loss = model(questions, labels)
            accelerator.backward(loss)
            optimizer.step()
            total_train_loss += loss.item() / len(train_dataloader)
        scheduler.step()
        accelerator.log({"train_loss": total_train_loss}, step=epoch)
        logger.info(f'Train Epoch {epoch}: Loss {round(total_train_loss, 4)}' + 
                    '\n------------------------------------------------------')
        
        total_val_loss = 0
        total_acc = 0
        if epoch % eval_interval == 0:
            for _, batch in enumerate(val_dataloader):
                model.eval()
                qaids, questions, metas, labels = batch
                loss, acc, outputs = model.inference(questions, labels)
                total_acc += acc.item() / len(val_dataloader)
                total_val_loss += loss.item() / len(val_dataloader)
            accelerator.log({"val_loss": total_val_loss,
                             "val_acc": total_acc}, step=epoch)
            logger.info(f'Eval Epoch {epoch}: Loss {round(total_val_loss, 4)}' + 
                        #'\n' + json.dumps(outputs, indent=2) + '\n' + 
                        #str(labels) +
                        '\n------------------------------------------------------')
        
            if total_val_loss < prev_val_loss and save_ckpt: 
                if not os.path.exists(save_ckpt_dir):
                    os.makedirs(save_ckpt_dir, exist_ok=True)
                save_ckpt_path = f'{save_ckpt_dir}/{epoch}'
                accelerator.save_model(model.pred_head, save_ckpt_path)
            prev_val_loss = total_val_loss
    
    
    
if __name__ == "__main__":
    fire.Fire(train_filter)
    
    