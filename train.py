from lightning.lightning_model import SSD300_COCO
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl


# TODO :- remove config and replace with cfg yaml file
# TODO :- make train working with scheduler and everything, logs too
# TODO :- see saving model to take up less space
# TODO :- make inference.py file/functions so that can work with it in validation_end() func, and calc AP

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    model = SSD300_COCO(cfg=config)

    print(config.pretty())

    trainer = pl.Trainer(train_percent_check=0.001, val_percent_check=0.001)
    trainer.fit(model)

    # data = COCODataset(config.train.train_data_path, config.train.train_annotate_path, 'TEST')
    # print(config.train.train_data_path)
    # if os.path.isfile(config.train.train_annotate_path):
    #     print('Yes')
    # for x in data:
    #     print(x)
    #     break

