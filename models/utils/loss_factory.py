import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    weighted_loss = alpha * loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss
class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, samples_per_class=None, beta=0.9999, gamma=0.5, loss_type="focal"):
        super(ClassBalancedLoss, self).__init__()
        if loss_type not in ["focal", "sigmoid", "softmax"]:
            loss_type = "focal"
        if samples_per_class is None:
            num_classes = 5000
            samples_per_class = [1] * num_classes
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights[samples_per_class == 0] = 0
        self.constant_sum = np.sum(samples_per_class!=0)
        weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)
        self.class_weights = weights
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type


    def update(self, samples_per_class):
        if samples_per_class is None:
            return
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        self.constant_sum = len(samples_per_class)
        weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)
        self.class_weights = weights



    def forward(self, x, y):
        _, num_classes = x.shape
        labels_one_hot = F.one_hot(y, num_classes).float()
        weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y)
        weights = weights.unsqueeze(1)
        if self.loss_type == "focal":
            cb_loss = focal_loss(x, labels_one_hot, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(x, labels_one_hot, weights)
        else:  # softmax
            pred = x.softmax(dim=1)
            # print(y, weights)
            cb_loss = F.binary_cross_entropy(pred, labels_one_hot, weights)
        return cb_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

