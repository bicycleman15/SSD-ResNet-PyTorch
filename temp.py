from priorbox import PriorBox
from config import config
priors = PriorBox(cfg = config)
import torch

priors = priors.create_priors()

from multibox import MultiBoxLoss
model = MultiBoxLoss(6,priors,config)
loc_preds = torch.rand(2, 5776, 4).cuda()
conf_preds = torch.rand(2, 5776, 6).cuda()

boxes = [torch.rand(6,4), torch.rand(2,4)]
labels = [torch.randint(5,(6,)), torch.randint(5,(2,))]

loc = model.forward(loc_preds,conf_preds,boxes,labels)

print(loc)