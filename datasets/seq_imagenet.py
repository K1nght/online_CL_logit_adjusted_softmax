import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from backbone.ResNet18 import resnet18

from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.targeted_image_folder import find_classes, TargetedImageFolder
from datasets.transforms.transforms_factory import create_transform
from datasets.transforms.denormalization import DeNormalize


class SequentialImagenet(ContinualDataset): 

    NAME = 'seq-imagenet'
    SETTING = 'class-il'
    N_CLASSES = 1000
    N_TASKS = 100
    N_CLASSES_PER_TASK = 10

    INPUT_SIZE = (3, 224, 224)
    TRANSFORM = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    AUG_TRANSFORM = create_transform(
        (224, 224),
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
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __init__(self, args):
        self.data_path = "/home/pami/Datasets/ILSVRC2012"
        self.classes, self.class_to_idx = find_classes(os.path.join(self.data_path, 'train'))
        self.target_classes_list = []
        self.target_classes_to_idx_list = []
        num_per_task = self.N_CLASSES // self.N_TASKS 

        for task in range(self.N_TASKS):
            target_classes = self.classes[task*num_per_task:(task+1)*num_per_task]
            target_classes_to_idx = [self.class_to_idx[c] for c in target_classes]
            self.target_classes_list.append(target_classes)
            self.target_classes_to_idx_list.append(target_classes_to_idx)
        self.CLASSES_PER_TASK = self.target_classes_to_idx_list
            
        super(SequentialImagenet, self).__init__(args)

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM 

        target_classes = self.target_classes_list[self.i]
        train_dataset = TargetedImageFolder(
            os.path.join(self.data_path, 'train'), target_classes, transform, return_path=True
        )
        test_dataset = TargetedImageFolder(
            os.path.join(self.data_path, 'val'), target_classes, test_transform
        )

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
        return resnet18(SequentialImagenet.N_CLASSES)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImagenet.TRANSFORM])
        return transform

    @staticmethod
    def get_aug_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImagenet.AUG_TRANSFORM])
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
        return SequentialImagenet.get_batch_size()



