from lightning.lightning_model import SSD300_COCO
import torch
from omegaconf import OmegaConf

# TODO :- remove config and replace with cfg yaml file
# TODO :- make train working with scheduler and everything, logs too
# TODO :- see saving model to take up less space
# TODO :- make inference.py file/functions so that can work with it in validation_end() func, and calc AP

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    model = SSD300_COCO(cfg=config)

    model.eval()
    with torch.no_grad():
        image = torch.rand(1,3,300,300)
    locs, confs = model(image)

    print(locs.shape, confs.shape)

