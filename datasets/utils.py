import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from itertools import product

import numpy as np
from torch.utils.data import DataLoader
from omegaconf import open_dict

from utils.constants import *
from datasets import LilacDataset


def download_dataset(cfg):
    url = f"https://www2.informatik.uni-hamburg.de/wtm/datasets2/{cfg.dataset}.zip"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with ZipFile(BytesIO(response.content)) as zfile:
        for data in response.iter_content(1024):
            progress_bar.update(len(data))
        zfile.extractall(cfg.data_dir)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Something went wrong while downloading the dataset!")


def init_tasks(cfg):
    # Get all tasks
    tasks = list(
        product(COLORS, OBJECT_TYPES, DIRECTIONS)
        if cfg.dataset.split("-")[-1] == "2d"
        else product(BLOCK_COLORS, SIZES, BOWL_COLORS)
    )
    init_tasks = INIT_TASKS[cfg.dataset.split("-")[-1]]
    continual_tasks = [t for t in tasks if t not in init_tasks]

    # Get the initialization and continual task indices
    indices = list(range(len(tasks)))
    init_indices = [i for i in indices if tasks[i] in init_tasks]
    continual_indices = [i for i in indices if i not in init_indices]

    with open_dict(cfg):
        cfg.init = {"indices": init_indices, "tasks": init_tasks}
        cfg.continual = {"indices": continual_indices, "tasks": continual_tasks}


def load_data(cfg):
    params = {
        "batch_size": cfg.batch_size,
        "drop_last": False,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
    }
    init_dl = DataLoader(
        dataset=LilacDataset(cfg, split=cfg.init, mode="train"), **params
    )

    # Create num_tasks dataloader for continual setup
    continual_dl = []
    eval_dl = [
        DataLoader(dataset=LilacDataset(cfg, split=cfg.init, mode="val"), **params)
    ]
    test_dl = [
        DataLoader(dataset=LilacDataset(cfg, split=cfg.init, mode="test"), **params)
    ]
    permuted_task_indices = np.random.permutation(60).reshape(10, -1)
    with open_dict(cfg):
        cfg.continual_tasks = {"indices": [], "tasks": []}

    for indices in permuted_task_indices:
        split = {
            "indices": [
                j for i, j in enumerate(cfg.continual["indices"]) if i in indices
            ],
            "tasks": [j for i, j in enumerate(cfg.continual["tasks"]) if i in indices],
        }
        with open_dict(cfg):
            cfg.continual_tasks["indices"].append(split["indices"])
            cfg.continual_tasks["tasks"].append(split["tasks"])

        continual_dl.append(
            DataLoader(dataset=LilacDataset(cfg, split=split, mode="train"), **params)
        )
        eval_dl.append(
            DataLoader(dataset=LilacDataset(cfg, split=split, mode="val"), **params)
        )
        test_dl.append(
            DataLoader(dataset=LilacDataset(cfg, split=split, mode="test"), **params)
        )

    return init_dl, continual_dl, eval_dl, test_dl
