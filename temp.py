from priorbox import PriorBox
from config import config
priors = PriorBox(cfg = config)
import torch

num_classes = config['num_classes'] # +1 for background class

priors = priors.create_priors()

from multibox import MultiBoxLoss
criterion = MultiBoxLoss(num_classes,priors,config)
if torch.cuda.is_available():
    criterion.cuda("cuda:1")

from model import SSD300

model = SSD300()

if torch.cuda.is_available():
    model.cuda("cuda:1")

from dataset import COCODataset

data = COCODataset('dataset/val2017','dataset/annotations/instances_val2017.json',split='TRAIN')
train_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True,
                                               collate_fn=data.collate_fn, num_workers=4)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.001)

from tqdm import tqdm

i = 0

for imgs, bboxs, labels in (train_loader):

    optimizer.zero_grad()

    imgs = imgs.cuda("cuda:1")
    locs, confs = model(imgs)

    loss = criterion.forward(locs, confs, bboxs, labels)

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.item())
    
    i += 1
