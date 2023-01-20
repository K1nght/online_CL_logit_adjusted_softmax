import numpy as np 
from collections import Counter

import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.ResNet import resnet34

from utils.conf import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset, get_train_loader_by_index, get_test_loader_by_index
from datasets.transforms.transforms_factory import create_transform
from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100

class BlurryCIFAR100(ContinualDataset):

    NAME = 'blurry-cifar100'
    SETTING = 'blurry-class-il'
    N_CLASSES = 100
    N_TASKS = -1
    N_CLASSES_PER_TASK = -1

    INPUT_SIZE = (3, 32, 32)
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])

    AUG_TRANSFORM = create_transform(
        INPUT_SIZE,
        use_prefetcher=False,
        scale=[0.75, 1.0],
        ratio=[3./4., 4./3.],
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bilinear',
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        re_num_splits=0,
        separate=False 
    )

    def __init__(self, args):
        self.CLASSES_PER_TASK = []
        super(BlurryCIFAR100, self).__init__(args)
        self.N_TASKS = args.blurry_n_tasks 
        self.datalist = []
        self.train_datasets = []
        self.test_transform = transforms.Compose(
                [transforms.ToTensor(), self.get_normalization_transform()])

        num_disjoint_cls = int(self.N_CLASSES*args.blurry_n/100)
        num_blurry_cls = self.N_CLASSES - num_disjoint_cls

        self.disjoint_cls = np.arange(self.N_CLASSES)[:num_disjoint_cls]
        self.blurry_cls = np.arange(self.N_CLASSES)[num_disjoint_cls:]

        self.train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=self.TRANSFORM)

        self.test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=self.test_transform)

        self.train_cls_indexs = []
        self.test_cls_indexs = []
        num_blurry_data = {}
        for label in range(self.N_CLASSES):
            train_cls_index = np.where(np.array(self.train_dataset.targets) == label)[0]
            test_cls_index = np.where(np.array(self.test_dataset.targets) == label)[0]
            if label in self.blurry_cls:
                np.random.shuffle(train_cls_index)
                num_blurry_data[label] = len(train_cls_index)
            # print(train_cls_index, label)
            self.train_cls_indexs.append(train_cls_index)
            self.test_cls_indexs.append(test_cls_index)
        
        self.train_data_indexs = []

        for t in range(self.N_TASKS):
            if t == self.N_TASKS-1:
                cur_disjoint_cls = self.disjoint_cls[t*int(num_disjoint_cls/self.N_TASKS):]
                cur_blurry_majority = self.blurry_cls[t*int(num_blurry_cls/self.N_TASKS):]
                cur_blurry_minority = self.blurry_cls[:t*int(num_blurry_cls/self.N_TASKS)]
            else:
                cur_disjoint_cls = \
                    self.disjoint_cls[t*int(num_disjoint_cls/self.N_TASKS):(t+1)*int(num_disjoint_cls/self.N_TASKS)]
                cur_blurry_majority = \
                    self.blurry_cls[t*int(num_blurry_cls/self.N_TASKS):(t+1)*int(num_blurry_cls/self.N_TASKS)]
                cur_blurry_minority = np.concatenate(
                    (
                        self.blurry_cls[:t*int(num_blurry_cls/self.N_TASKS)], 
                        self.blurry_cls[(t+1)*int(num_blurry_cls/self.N_TASKS):]
                    )
                )
            # print(cur_disjoint_cls, cur_blurry_majority, cur_blurry_minority)
            self.CLASSES_PER_TASK.append(np.concatenate((cur_disjoint_cls, self.blurry_cls)))
            cur_train_data_index = []
            # disjoint class
            for label in cur_disjoint_cls:
                cur_train_data_index.append(self.train_cls_indexs[label])
            # blurry class 
            for label in cur_blurry_majority:
                num_data = num_blurry_data[label] - (self.N_TASKS-1) * args.blurry_m
                cur_train_data_index.append(self.train_cls_indexs[label][:num_data])
                self.train_cls_indexs[label] = self.train_cls_indexs[label][num_data:]
            for label in cur_blurry_minority:
                num_data = args.blurry_m
                cur_train_data_index.append(self.train_cls_indexs[label][:num_data])
                self.train_cls_indexs[label] = self.train_cls_indexs[label][num_data:]
            self.train_data_indexs.append(np.concatenate(cur_train_data_index))
        
        # print(self.CLASSES_PER_TASK)
        # print([Counter([self.train_dataset.targets[i] for i in c]) for c in self.train_data_indexs])

        self.cls_seen_so_far = None 
        for classes in self.CLASSES_PER_TASK:
            self.cls_seen_so_far = set(classes) if self.cls_seen_so_far is None else (self.cls_seen_so_far & set(classes))
            # print(self.cls_seen_so_far)
        # print(f"start cls", self.cls_seen_so_far)

        # if len(self.cls_seen_so_far) > 0:
        #     for cls in self.cls_seen_so_far:
        #         cls_index = self.test_cls_indexs[cls]
        #         init_test_loader = get_test_loader_by_index(self.test_dataset, cls_index, self)
        #         self.test_loaders.append(init_test_loader)


    def get_data_loaders(self):
        train_data_index = self.train_data_indexs[self.i]

        train_loader = get_train_loader_by_index(self.train_dataset, train_data_index, self)

        self.train_loader = train_loader 

        self.cls_seen_so_far = self.cls_seen_so_far | set(self.CLASSES_PER_TASK[self.i]) 

        test_cls_index = np.concatenate([self.test_cls_indexs[cls] for cls in self.cls_seen_so_far])
        
        test_loader = get_test_loader_by_index(self.test_dataset, test_cls_index, self)

        self.test_loaders = [test_loader]

        self.i += 1 
        return train_loader, test_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), BlurryCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_aug_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), BlurryCIFAR100.AUG_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet34(BlurryCIFAR100.N_CLASSES)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return BlurryCIFAR100.get_batch_size()
