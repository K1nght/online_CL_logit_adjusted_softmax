from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path
from datasets.transforms.transforms_factory import create_transform

class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_TASKS = 10
    N_CLASSES_PER_TASK = 10

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
        self.CLASSES_PER_TASK = [
            [t*(self.N_CLASSES//self.N_TASKS)+i for i in range(self.N_CLASSES//self.N_TASKS)] 
            for t in range(self.N_TASKS)]
        super(SequentialCIFAR100, self).__init__(args)

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_aug_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.AUG_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES)

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
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    # @staticmethod
    # def get_scheduler(model, args) -> torch.optim.lr_scheduler:
    #     model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
    #     return scheduler

    @staticmethod
    def get_scheduler(model, args):
        return None

