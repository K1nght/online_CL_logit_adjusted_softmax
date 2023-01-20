import numpy
import importlib
import os
import socket
import sys
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

our_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(our_path)
sys.path.append(our_path + '/datasets')
sys.path.append(our_path + '/backbone')
sys.path.append(our_path + '/models')

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model

from utils.args import add_management_args
from utils.conf import set_random_seed
from utils.distributed import make_dp
from utils.training import train


def parse_args():
    parser = ArgumentParser(description='online CL', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    # torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.distributed == 'dp':
        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        args.nowand = 1

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    train(model, dataset, args)


if __name__ == '__main__':
    main()
