import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


class CustomDP(DataParallel):

    intercept_names = ['classifier', 'num_classes', 'set_return_prerelu']

    def __getattr__(self, name: str):
        if name in self.intercept_names:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.intercept_names:
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)

def make_dp(model):
    return CustomDP(model, device_ids=range(torch.cuda.device_count())).to('cuda:0')
