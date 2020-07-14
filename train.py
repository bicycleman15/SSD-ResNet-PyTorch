from priorbox import PriorBox
from config import config
priors = PriorBox(cfg = config)
import torch

num_classes = config['num_classes'] # +1 for background class

priors = priors.create_priors()

from multibox import MultiBoxLoss
criterion = MultiBoxLoss(num_classes,priors,config)
if torch.cuda.is_available():
    criterion.cuda()

from model import SSD300

model = SSD300()

if torch.cuda.is_available():
    model.cuda()

from dataset import COCODataset

data = COCODataset('../val2017','../annotations/instances_val2017.json')
train_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True,
                                               collate_fn=data.collate_fn, num_workers=4)

for imgs, bboxs, labels in train_loader:
    print(imgs.shape)
    imgs = imgs.cuda()
    locs, confs = model(imgs)
    
    print(locs.shape)
    print(confs.shape)

    loss = criterion.forward(locs, confs, bboxs, labels)
    print(loss)
    break