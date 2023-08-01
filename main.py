import argparse
import logging
import os
import random
import socket
import sys
import yaml

import numpy as np
import psutil
import setproctitle
import torch
from utils.record import *
from default import cfg



from baseFed.FedManager import FedManager

setproctitle.setproctitle("FedAvg")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))



def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    

    torch.backends.cudnn.deterministic =True
    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str)
    args = parser.parse_args()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    
    logging_config(cfg,"FedAvg")
    
    if cfg.wandb_record and cfg.record_tool == 'wandb':
        import wandb
        wandb.init(project=cfg.wandb_project,
            name='test',
            config=dict(cfg)
            )
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    set_random(cfg.seed)

    fedmanager = FedManager(device, cfg)
    fedmanager.train()