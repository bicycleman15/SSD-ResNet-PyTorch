from priorbox import PriorBox
from config import config
priors = PriorBox(cfg = config)
import torch

device = config['device']

priors = priors.create_priors()
priors = priors.to(device)

from multibox import MultiBoxLoss
criterion = MultiBoxLoss(priors,config)
criterion = criterion.to(device)

from model import SSD300

model = SSD300()
model = model.to(device)

from dataset import COCODataset

data = COCODataset('../val2017','../annotations/instances_val2017.json',split='TEST')
train_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True,
                                               collate_fn=data.collate_fn, num_workers=4)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.1)

from tqdm import tqdm

images, bboxes, labels = next(iter(train_loader))


# print(len(bboxes))
# print(images.shape)

images = images.to(device)
bboxes = [x.cuda() for x in  bboxes] 
labels = [x.cuda() for x in labels]

# bboxes = [torch.zeros((0,4)).cuda(), torch.zeros((0,4)).cuda()]
# labels = [torch.zeros(0,91).cuda(), torch.zeros(0,91).cuda()]
with torch.no_grad():
    locs, confs = model(images)
# loc_loss, conf_loss = criterion(locs, confs, bboxes, labels)

from detect_utils import Detect
detection = Detect(config)

boxes = detection.forward(locs, confs, priors)
print(boxes.shape)

import pdb
pdb.set_trace()

# print(loc_loss, conf_loss)
# loss = loc_loss + conf_loss

# print(loss)

# i = 0

# for imgs, bboxs, labels in (train_loader):

#     optimizer.zero_grad()

#     imgs = imgs.to(device)
#     bboxs = [box.to(device) for box in bboxs]
#     labels = [label.to(device) for label in labels]

#     locs, confs = model(imgs)

#     loc_loss, conf_loss = criterion(locs, confs, bboxs, labels)
#     loss = loc_loss + conf_loss
#     loss.backward()
#     optimizer.step()

#     if i % 10 == 0:
#         print(loss.item())
    
#     i += 1