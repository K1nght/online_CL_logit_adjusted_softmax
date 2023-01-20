from collections import deque
import numpy as np 
import torch

from models.utils.continual_model import ContinualModel
from models.utils.aser_utils import ClassBalancedRandomSampling, add_minority_class_input, compute_knn_sv, nonzero_indices
from datasets.utils.constant import n_classes
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ASER with Logit Adjusted Softmax')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--k', dest='k', default=5, type=int,
                    help='Number of nearest neighbors (K) to perform ASER (default: %(default)s)')
    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                    help='Type of ASER: '
                            '"neg_sv" - Use negative SV only,'
                            ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                            ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')
    parser.add_argument('--n_smp_cls', dest='n_smp_cls', default=6.0, type=float,
                    help='Maximum number of samples per class for random sampling (default: %(default)s)')
    parser.add_argument('--tau', type=float, default=1.,
                    help='adjustment param.')
    parser.add_argument('--window_length', type=int, default=1,
                        help='sliding window length.')
    return parser


class ASER_update(object):
    def __init__(self, args, device):
        super().__init__() 
        self.device = device 
        self.args = args 
        self.k = args.k 
        self.buffer_size = args.buffer_size 
        self.out_dim = n_classes[args.dataset]
        self.n_smp_cls = int(args.n_smp_cls)
        self.n_total_smp = int(args.n_smp_cls * self.out_dim)
        ClassBalancedRandomSampling.class_index_cache = None

    def update(self, net, buffer, x, y, transform):

        place_left = self.buffer_size - len(buffer)

        if place_left:
            x_fit = x[:place_left]
            y_fit = y[:place_left]

            ind = torch.arange(start=len(buffer), end=len(buffer)+x_fit.size(0))
            ClassBalancedRandomSampling.update_cache(buffer.labels, self.out_dim,
                                                     new_y=y_fit, ind=ind)
            # reservoir update
            buffer.add_data(examples=x, labels=y)
            
        if len(buffer) == self.buffer_size:
            cur_x, cur_y = x[place_left:], y[place_left:]
            self._update_by_knn_sv(net, buffer, cur_x, cur_y, transform)

    def _update_by_knn_sv(self, net, buffer, cur_x, cur_y, transform):
        """
            Returns indices for replacement.
            Buffered instances with smallest SV are replaced by current input with higher SV.
                Args:
                    model (object): neural network.
                    buffer (object): buffer object.
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                Returns
                    ind_buffer (tensor): indices of buffered instances to be replaced.
                    ind_cur (tensor): indices of current data to do replacement.
        """
        cur_x = cur_x.to(self.device)
        cur_y = cur_y.to(self.device)

        # Find minority class samples from current input batch
        minority_batch_x, minority_batch_y = add_minority_class_input(cur_x, cur_y, self.buffer_size, self.out_dim)

        # Evaluation set
        eval_x, eval_y, eval_indices = \
        ClassBalancedRandomSampling.sample(buffer, self.n_smp_cls)

        # Concatenate minority class samples from current input batch to evaluation set
        eval_x = torch.cat((eval_x, minority_batch_x))
        eval_y = torch.cat((eval_y, minority_batch_y))

        # Candidate set
        cand_excl_indices = set(eval_indices.tolist())
        filled_indices = np.arange(len(buffer))
        valid_indices = np.setdiff1d(filled_indices, np.array(cand_excl_indices))
        num_retrieve = min(self.n_total_smp, valid_indices.shape[0])
        cand_ind = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, replace=False)).long()
        cand_x, cand_y = buffer.get_data_by_index(cand_ind)

        eval_x = torch.stack([transform(x.cpu()) for x in eval_x]).to(self.device)
        cand_x = torch.stack([transform(x.cpu()) for x in cand_x]).to(self.device)
        sv_matrix = compute_knn_sv(net, eval_x, eval_y, cand_x, cand_y, self.k, device=self.device)
        sv = sv_matrix.sum(0)

        n_cur = cur_x.size(0)
        n_cand = cand_x.size(0)

        # Number of previously buffered instances in candidate set
        n_cand_buf = n_cand - n_cur

        sv_arg_sort = sv.argsort(descending=True)
        # Divide SV array into two segments
        # - large: candidate args to be retained; small: candidate args to be discarded
        sv_arg_large = sv_arg_sort[:n_cand_buf]
        sv_arg_small = sv_arg_sort[n_cand_buf:]

        # Extract args relevant to replacement operation
        # If current data instances are in 'large' segment, they are added to buffer
        # If buffered instances are in 'small' segment, they are discarded from buffer
        # Replacement happens between these two sets
        # Retrieve original indices from candidate args
        ind_cur = sv_arg_large[nonzero_indices(sv_arg_large >= n_cand_buf)] - n_cand_buf
        arg_buffer = sv_arg_small[nonzero_indices(sv_arg_small < n_cand_buf)]
        ind_buffer = cand_ind[arg_buffer]

        # perform overwrite op 
        x_upt = cur_x[ind_cur]
        y_upt = cur_y[ind_cur]
        ClassBalancedRandomSampling.update_cache(buffer.labels, self.out_dim,
                                                 new_y=y_upt, ind=ind_buffer)

        for i, idx in enumerate(ind_buffer):
            buffer.examples[idx] = x_upt[i] 
            buffer.labels[idx] = y_upt[i]

class ASER_retrieve(object):
    def __init__(self, args, device):
        super().__init__() 
        self.num_retrieve = args.minibatch_size 
        self.args = args 
        self.device = device 
        self.k = args.k 
        self.buffer_size = args.buffer_size 
        self.aser_type = args.aser_type 
        self.n_smp_cls = int(args.n_smp_cls)
        self.out_dim = n_classes[args.dataset]
        ClassBalancedRandomSampling.class_index_cache = None

    def retrieve(self, net, buffer, transform, cur_x, cur_y):
        if len(buffer) < self.buffer_size:
            # Use random retrieval until buffer is filled
            ret_x, ret_y = buffer.get_data(
                self.num_retrieve, transform=transform)
        else:
            # Use ASER retrieval if buffer is filled
            ret_x, ret_y = self._retrieve_by_knn_sv(net, buffer, cur_x, cur_y, self.num_retrieve, transform)
        return ret_x, ret_y 
    
    def _retrieve_by_knn_sv(self, net, buffer, cur_x, cur_y, num_retrieve, transform):
        """
            Retrieves data instances with top-N Shapley Values from candidate set.
                Args:
                    net (object): neural network.
                    buffer.
                    cur_x (tensor): current input data tensor.
                    cur_y (tensor): current input label tensor.
                    num_retrieve (int): number of data instances to be retrieved.
                Returns
                    ret_x (tensor): retrieved data tensor.
                    ret_y (tensor): retrieved label tensor.
        """
        cur_x = cur_x.to(self.device)
        cur_y = cur_y.to(self.device)

        # Get candidate data for retrieval (i.e., cand <- class balanced subsamples from memory)
        cand_x, cand_y, cand_ind = \
            ClassBalancedRandomSampling.sample(buffer, self.n_smp_cls, transform=transform)

        # Type 1 - Adversarial SV 
        # Get evaluation data for type 1 (i.e., eval <- current input)
        eval_adv_x, eval_adv_y = cur_x, cur_y 
        # Compute adversarial Shapley value of candidate data 
        # (i.e., sv wrt current input)
        sv_matrix_adv = compute_knn_sv(net, eval_adv_x, eval_adv_y, cand_x, cand_y, self.k, device=self.device)

        if self.aser_type != "neg_sv":
            # Type 2 - Cooperative SV 
            # Get evaluation data for type 2 
            # (i.e., eval <- class balanced subsamples from memory excluding those already in candidate set)
            excl_indices = set(cand_ind.tolist())
            eval_coop_x, eval_coop_y, _ = \
                ClassBalancedRandomSampling.sample(buffer, self.n_smp_cls, excl_indices=excl_indices)

            # Compute Shapley value 
            sv_matrix_coop = \
                compute_knn_sv(net, eval_coop_x, eval_coop_y, cand_x, cand_y, self.k, device=self.device)
            if self.aser_type == "asv":
                # Use extremal SVs for computation 
                sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values 
            else:
                # Use mean variation for aser_type == "asvm" or anything else 
                sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)
        else:
            # aser_type == "neg_sv"
            # No Type 2 - Cooperative SV; Use sum of Adversarial SV only 
            sv = sv_matrix_adv.sum(0) * (-1)
        
        ret_ind = sv.argsort(descending=True)

        ret_x = cand_x[ret_ind][:num_retrieve]
        ret_y = cand_y[ret_ind][:num_retrieve]
        return ret_x, ret_y


class ASERLAS(ContinualModel):
    NAME = 'aser_las'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ASERLAS, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.update_method = ASER_update(self.args, self.device)
        self.retrieve_method = ASER_retrieve(self.args, self.device)
        self.test_transform = None
        self.num_classes = n_classes[self.args.dataset]
        self.label_freq_record = torch.zeros((1, self.num_classes)).to(self.device)
        self.label_freq_deque = deque()

    def begin_task(self, dataset):
        self.test_transform = dataset.get_normalization_transform()

    def observe(self, inputs, labels, not_aug_inputs):
        
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = \
                self.retrieve_method.retrieve(self.net, self.buffer, self.transform, inputs, labels)
            cat_inputs = torch.cat((inputs, buf_inputs))
            cat_labels = torch.cat((labels, buf_labels))
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

        if not hasattr(self.buffer, 'examples'):
            self.buffer.init_tensors(not_aug_inputs, labels, None, None)

        self.update_method.update(self.net, self.buffer, not_aug_inputs,
                                  labels, self.test_transform)

        return loss.item()
