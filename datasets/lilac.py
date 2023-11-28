import json
import os
import random
import torch
from torchvision import io
from torch.utils.data import Dataset


class LilacDataset(Dataset):
    def __init__(self, cfg, split, mode="train"):
        self.root_dir = os.path.join(cfg.data_dir, cfg.dataset)
        with open(os.path.join(self.root_dir, f"{mode}.json")) as file:
            self.data = json.load(file)
        self.data = [d for d in self.data if d["task"] in split["indices"]]
        img_path = os.path.join(self.root_dir, "images", mode)
        self.images = [
            (io.read_image(os.path.join(img_path, file["images"][i])) / 256)
            for file in self.data
            for i in range(3)
        ]
        self.images = [self.images[i : i + 3] for i in range(0, len(self.images), 3)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mission = torch.tensor(self.data[idx]["mission2idx"])
        task = torch.tensor(self.data[idx]["task"])

        # base_img, pos_img, neg_img, mission, concept being challenged with neg_img, task descriptor
        return (
            self.images[idx][0],
            self.images[idx][1],
            self.images[idx][2],
            mission,
            task,
        )
