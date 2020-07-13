import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, priors, cfg, overlap_thresh = 0.5, neg_pos = 3):
        super(MultiBoxLoss, self).__init__()

        self.num_classes = num_classes # Total classes + 1 for background
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = cfg['variance']
        self.priors = priors

        self.alpha = 1.0

    def forward(self, loc_preds, conf_preds, boxes, labels):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        priors = self.priors
        num = loc_preds.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.zeros((num, num_priors, 4), dtype=torch.float, requires_grad=False)
        conf_t = torch.zeros((num, num_priors), dtype=torch.long, requires_grad=False)

        for idx in range(num):  # CPU calc below
            loc_t[idx], conf_t[idx] = match(self.threshold, boxes[idx].data, priors.data, self.variance, labels[idx].data)
        
        if torch.cuda.is_available():
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos_priors = conf_t > 0 # [num, 8732]

        loc_loss = F.smooth_l1_loss(loc_preds[pos_priors], loc_t[pos_priors], reduction='mean')

        n_pos = pos_priors.sum(dim=1) # [num]
        n_hard_negs = self.negpos_ratio * n_pos

        conf_loss_all = F.cross_entropy(conf_preds.view(-1,self.num_classes), conf_t.view(-1), reduction='none')
        
        conf_loss_all = conf_loss_all.view(num, num_priors)

        conf_loss_pos = conf_loss_all[pos_priors]

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[pos_priors] = 0. # ignore pos priors

        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) # [num,8732]
        
        hardness_ranks = torch.LongTensor(range(num_priors)).unsqueeze(0).expand_as(conf_loss_neg)  # (num, 8732)
        if torch.cuda.is_available():
            hardness_ranks = hardness_ranks.cuda()
        hard_negatives = hardness_ranks < n_hard_negs.unsqueeze(1)  # (num, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negs))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = conf_loss_hard_neg.sum() + conf_loss_pos.sum()
        N = (n_pos.sum().float() + 1e-7) 

        conf_loss /= N

        # TOTAL LOSS  = L(x,c,l,g) = (Lconf(x, c) + α * Lloc(x,l,g)) / N
        return conf_loss + self.alpha * loc_loss

