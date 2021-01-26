import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")



@hydra.main(config_path="config", config_name="default")
def main(args):
    # Setting the random seed throughout the modules
    if args.seed is not None:
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
    args.env_args['seed'] = args.seed

    config = OmegaConf.to_container(args)
    if args.use_wandb:
        wandb.login()
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group, config=config)

    # run the framework
    logger = get_logger()
    run(config, logger)

if __name__ == '__main__':
    main()

