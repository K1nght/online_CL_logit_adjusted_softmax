import os
import json 
from typing import Optional

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from backbone.ResNet import resnet18

from datasets.utils.continual_dataset import ContinualDataset
from datasets.transforms.transforms_factory import create_transform
from datasets.transforms.denormalization import DeNormalize


def default_loader(path):
    return Image.open(path).convert('RGB')


class INAT(Dataset):
    def __init__(self, root, imgs, targets, transform=None, target_transform=None,
                 loader=default_loader, return_path=False):
        self.root = root 
        self.imgs = imgs 
        self.targets = targets 
        self.transform = transform 
        self.target_transform = target_transform 
        self.loader = loader 
        self.return_path = return_path
    
    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        target = self.targets[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return img, target, path
        return img, target

    def __len__(self):
        return len(self.imgs)


class SequentialINaturalist(ContinualDataset): 

    NAME = 'seq-inat'
    SETTING = 'class-il'
    N_CLASSES = 5089
    N_TASKS = 26
    N_CLASSES_PER_TASK = -1

    INPUT_SIZE = (3, 299, 299)
    TRANSFORM = transforms.Compose([
            transforms.RandomResizedCrop((299, 299), scale=(0.75, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    AUG_TRANSFORM = create_transform(
        (299, 299),
        use_prefetcher=False,
        scale=[0.75, 1.0],
        ratio=[3./4., 4./3.],
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        re_num_splits=0,
        separate=False
    )

    TEST_TRANSFORM = transforms.Compose([
            transforms.Resize(int(299/0.875)),
            transforms.CenterCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __init__(self, args):
        self.data_path = "/home/pami/Datasets/iNaturalist"
        self.train_file = os.path.join(self.data_path, "train2017.json")
        self.val_file = os.path.join(self.data_path, "val2017.json")

        with open(self.train_file) as data_file:
            self.train_ann_data = json.load(data_file)
        with open(self.val_file) as data_file:
            self.test_ann_data = json.load(data_file)

        self.idx_to_class = {cc['id']: cc['name'] for cc in self.train_ann_data['categories']}
        self.class_to_idx = {cc['name']: cc['id'] for cc in self.train_ann_data['categories']}

        self.train_imgs_per_task = [[] for _ in range(self.N_TASKS)]
        self.train_targets_per_task = [[] for _ in range(self.N_TASKS)]
        for ann_img, ann_cls in zip(self.train_ann_data['images'], self.train_ann_data['annotations']):
            img = ann_img['file_name']
            cls = ann_cls['category_id']
            task = ord(self.idx_to_class[cls][0]) - ord("A")
            self.train_imgs_per_task[task].append(img)
            self.train_targets_per_task[task].append(cls)
        self.CLASSES_PER_TASK = [list(set(ts)) for ts in self.train_targets_per_task]
        
        self.test_imgs_per_task = [[] for _ in range(self.N_TASKS)]
        self.test_targets_per_task = [[] for _ in range(self.N_TASKS)]
        for ann_img, ann_cls in zip(self.test_ann_data['images'], self.test_ann_data['annotations']):
            img = ann_img['file_name']
            cls = ann_cls['category_id']
            task = ord(self.idx_to_class[cls][0]) - ord("A")
            self.test_imgs_per_task[task].append(img)
            self.test_targets_per_task[task].append(cls)
            
        super(SequentialINaturalist, self).__init__(args)

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM 

        train_imgs = self.train_imgs_per_task[self.i]
        train_targets = self.train_targets_per_task[self.i]
        test_imgs = self.test_imgs_per_task[self.i]
        test_targets = self.test_targets_per_task[self.i]

        train_dataset = INAT(self.data_path, train_imgs, train_targets, transform=transform, return_path=True)
        test_dataset = INAT(self.data_path, test_imgs, test_targets, transform=test_transform)

        train_loader = DataLoader(train_dataset,
                              batch_size=self.args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset,
                                batch_size=self.args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader

        self.i += 1
        return train_loader, test_loader

    @staticmethod
    def get_backbone():
        net = resnet18(pretrained=True, num_classes=1000)
        net.fc = nn.Linear(net.embedding_dim, SequentialINaturalist.N_CLASSES)
        return net

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialINaturalist.TRANSFORM])
        return transform

    @staticmethod
    def get_aug_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialINaturalist.AUG_TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225),)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225),)
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_batch_size():
        return 256

    @staticmethod
    def get_minibatch_size():
        return SequentialINaturalist.get_batch_size()



