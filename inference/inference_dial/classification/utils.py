
import logging
import os
from datetime import datetime
import torch.distributed as dist

class Logger:
    def __init__(self, fn):
        
        self.rank = dist.get_rank()
        now = datetime.now()
        fn=f'logs/{now.strftime("%Y-%m-%d")}/{now.strftime("%H:%M")}_{fn}'
        if not os.path.exists(fn):
            os.makedirs(fn, exist_ok=True)
        logging.basicConfig(
            filename=f'{fn}/{self.rank}.log',
            filemode='w',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO,
            datefmt='%m/%d/%Y %I:%M:%S %p',
        )
        
        self.logger = logging.getLogger(__name__)
    
    def info(self, text):
        self.logger.info(text)

    

def print_0(text1, text2=None):
    if dist.get_rank() == 0:
        if text2 is None:
            print(text1)
        else:
            print(text1, text2)