import torch
import random
from omegaconf import open_dict
import numpy as np


def seed(cfg):
    # Set seed for all randomness sources (don't forget to set PYTHONHASHSEED as env variable)
    if cfg.seed == 0:
        with open_dict(cfg):
            cfg.seed = np.random.randint(10000)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    return "{:,}".format(num)


# Log count of trainable model params
def log_network_summary(cfg, model, log):
    log.info("======= Overview model parameters =======")
    param_count_total = count_parameters(model)
    log.info(f"Total number of model parameters: {format_number(param_count_total)}")
    param_count_base = count_parameters(model.encoder_decoder)
    log.info(
        f"    Encoder/project parameters (only trained in init phase): {format_number(param_count_base)}"
    )
    param_count_shared = count_parameters(model.shared)
    log.info(
        f"    Shared parameters (shared across tasks): {format_number(param_count_shared)}"
    )
    param_count_split = count_parameters(model.specialized)
    log.info(
        f"    Task-specific parameters (total): {format_number(param_count_split)}"
    )
    log.info(
        f"    Task-specific parameters (per 1 task): {format_number(int(param_count_split / 10))}"
    )
    with open_dict(cfg):
        cfg.param_count_total = param_count_total
        cfg.param_count_base = param_count_base
        cfg.param_count_shared = param_count_shared
        cfg.param_count_per_task = param_count_split / 10
