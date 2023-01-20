from argparse import Namespace
from typing import Tuple
from PIL import Image

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES: int
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    CLASSES_PER_TASK: list

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        if not all((self.NAME, self.SETTING, self.N_CLASSES, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError

class MyTrainDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, data, targets, transform = None, target_transform = None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data = data 
        self.targets = targets 
        self.transform = transform 
        self.target_transform = target_transform 


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def __len__(self):
        return len(self.data)

class MyTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, data, targets, transform = None, target_transform = None) -> None:
        self.data = data 
        self.targets = targets 
        self.transform = transform 
        self.target_transform = target_transform 


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target

    def __len__(self):
        return len(self.data)

def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.zeros_like(train_dataset.targets)
    test_mask = np.zeros_like(test_dataset.targets)
    for c in setting.CLASSES_PER_TASK[setting.i]:
        train_mask[np.array(train_dataset.targets) == c] = 1
        test_mask[np.array(test_dataset.targets) == c] = 1
    train_mask = train_mask == 1
    test_mask = test_mask == 1

    train_data = train_dataset.data[train_mask]
    test_data = test_dataset.data[test_mask]

    train_targets = np.array(train_dataset.targets)[train_mask]
    test_targets = np.array(test_dataset.targets)[test_mask]

    split_train_dataset = MyTrainDataset(train_data, train_targets, transform=train_dataset.transform)
    split_test_dataset = MyTestDataset(test_data, test_targets, transform=test_dataset.transform)

    train_loader = DataLoader(split_train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(split_test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader

def get_train_loader_by_index(train_dataset, index, setting):
    train_data = train_dataset.data[index]

    train_targets = np.array(train_dataset.targets)[index]

    split_train_dataset = MyTrainDataset(train_data, train_targets, transform=train_dataset.transform)

    train_loader = DataLoader(split_train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    return train_loader

def get_test_loader_by_index(test_dataset, index, setting):
    test_data = test_dataset.data[index]

    test_targets = np.array(test_dataset.targets)[index]

    split_test_dataset = MyTestDataset(test_data, test_targets, transform=test_dataset.transform)

    test_loader = DataLoader(split_test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)

    return test_loader