import torch

from multibox import MultiBoxLoss

model = MultiBoxLoss()

print(model)

model = model.cuda()

batch_size = 2
priors = 8732
predicted_locs = torch.rand(batch_size, priors, 4).cuda()
predicted_cla = torch.rand(batch_size, priors, 4).cuda()

boxes = [torch.rand(2,4).cuda(), torch.rand(3,4).cuda()]
labels = [torch.LongTensor([1,2]).cuda(), torch.LongTensor([1,2,3]).cuda()]

x = print(model.forward(predicted_locs, predicted_cla, boxes, labels))