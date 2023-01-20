from collections import deque

import torch

from models.utils.continual_model import ContinualModel
from datasets.utils.constant import n_classes
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Experience Replay with Logit Adjusted Softmax')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--tau', type=float, default=1.,
                        help='adjustment param.')
    parser.add_argument('--window_length', type=int, default=1,
                        help='sliding window length.')
    return parser


class ERLAS(ContinualModel):
    NAME = 'er_las'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ERLAS, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = n_classes[self.args.dataset]
        self.label_freq_record = torch.zeros((1, self.num_classes)).to(self.device)
        self.label_freq_deque = deque()

    def observe(self, inputs, labels, not_aug_inputs):
        if not self.buffer.is_empty():
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            cat_inputs = torch.cat((inputs, buf_inputs))
            cat_labels = torch.cat((labels, buf_labels))
        else:
            cat_inputs = inputs 
            cat_labels = labels

        present = cat_labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        label_freq = torch.zeros((1, self.num_classes)).to(self.device)
        for l in cat_labels: label_freq[:, l] += 1

        self.label_freq_record += label_freq 
        self.label_freq_deque.append(label_freq)
        while len(self.label_freq_deque) > self.args.window_length:
            self.label_freq_record -= self.label_freq_deque.popleft()

        adjustments = self.args.tau * torch.log((self.label_freq_record) + 1e-50)
        logits = self.net(cat_inputs) + adjustments

        self.opt.zero_grad()

        loss = self.loss(logits, cat_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()
