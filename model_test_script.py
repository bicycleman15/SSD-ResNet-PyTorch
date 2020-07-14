from model import SSD300
import torch

net=SSD300()
sample=torch.rand(2,3,300,300)

loc,conf=net(sample)

print(loc.shape)
print(conf.shape)

