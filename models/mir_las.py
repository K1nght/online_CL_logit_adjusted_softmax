import copy 
from collections import deque

import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from datasets.utils.constant import n_classes
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='MIR with Logit Adjusted Softmax')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--subsample', type=int, default=160,
                    help='for subsampling in replay.')
    parser.add_argument('--tau', type=float, default=1.,
                    help='adjustment param.')
    parser.add_argument('--window_length', type=int, default=1,
                        help='sliding window length.')
    return parser

def get_grad_vector(pp, grad_dims, device):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims)).to(device)

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1

def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net = copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters, grad_vector, grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data=param.data - lr*param.grad.data
    return new_net

class MIRLAS(ContinualModel):
    NAME = 'mir_las'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(MIRLAS, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.num_classes = n_classes[self.args.dataset]
        self.label_freq_record = torch.zeros((1, self.num_classes)).to(self.device)
        self.label_freq_deque = deque()

    def end_task(self, dataset):
        for data in dataset.train_loader:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, not_aug_inputs, logits = data
            else:
                inputs, labels, not_aug_inputs = data

            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels)

    def observe(self, inputs, labels, not_aug_inputs):

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        self.opt.zero_grad()
        loss.backward() 

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.subsample, transform=self.transform)
            grad_dims = []
            for param in self.net.parameters():
                grad_dims.append(param.data.numel())
            grad_vector = get_grad_vector(self.net.parameters, grad_dims, self.device)
            model_temp = get_future_step_parameters(self.net, grad_vector, grad_dims, lr=self.args.lr)

            with torch.no_grad():
                step = 0 
                pre_loss = []
                post_loss = []
                sub_batch_size = 2*self.args.batch_size
                while step*sub_batch_size < len(buf_inputs):
                    batch_inputs = buf_inputs[step*sub_batch_size:min((step+1)*sub_batch_size, len(buf_inputs))]
                    batch_labels = buf_labels[step*sub_batch_size:min((step+1)*sub_batch_size, len(buf_inputs))]
                    step += 1
                    logits_track_pre = self.net(batch_inputs)
                    logits_track_post = model_temp(batch_inputs)

                    pre_loss.append(F.cross_entropy(logits_track_pre, batch_labels, reduction="none"))
                    post_loss.append(F.cross_entropy(logits_track_post, batch_labels, reduction="none"))
                pre_loss = torch.cat(pre_loss, dim=0)
                post_loss = torch.cat(post_loss, dim=0)
                scores = post_loss - pre_loss 
                big_ind = scores.sort(descending=True)[1][:self.args.minibatch_size]

            retr_buf_inputs, retr_buf_labels = buf_inputs[big_ind], buf_labels[big_ind]
            cat_inputs = torch.cat((inputs, retr_buf_inputs))
            cat_labels = torch.cat((labels, retr_buf_labels))
        else:
            cat_inputs = inputs 
            cat_labels = labels

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

        return loss.item()

