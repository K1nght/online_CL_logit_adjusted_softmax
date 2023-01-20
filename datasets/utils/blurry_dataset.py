import os
from typing import List, Tuple

import PIL
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset


def get_train_datalist(dataset, n_tasks, m, n, rnd_seed, cur_iter: int) -> List:
    if n == 100 or m == 0:
        n = 100
        m = 0
    return pd.read_json(
        f"collections/{dataset}/{dataset}_split{n_tasks}_n{n}_m{m}_rand{rnd_seed}_task{cur_iter}.json"
    ).to_dict(orient="records")

def get_test_datalist(dataset) -> List:
    return pd.read_json(f"collections/{dataset}/{dataset}_val.json").to_dict(orient="records")


class StreamDataset(Dataset):
    def __init__(self, datalist, dataset, transform, data_dir=None, is_train=True) -> None:
        self.images = []
        self.labels = []
        self.dataset = dataset 
        self.data_dir = data_dir 
        self.transform = transform 
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.is_train = is_train

        for data in datalist: 
            try: 
                img_name = data['file_name']
            except KeyError: 
                img_name = data['filepath']
            if self.data_dir is None: 
                img_path = os.path.join("data", f"{self.dataset}_png", img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            self.images.append(PIL.Image.open(img_path).convert('RGB'))
            self.labels.append(data['label'])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> Tuple[Image.Image, int, Image.Image]:
        img, label = self.images[index], self.labels[index]

        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)
        if self.transform is not None:
            img = self.transform(img)

        if not self.is_train:
            return img, label
            
        if hasattr(self, 'logits'):
            return img, label, not_aug_img, self.logits[index]

        return img, label, not_aug_img
