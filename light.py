import torch
import pytorch_lightning as pl
from model import SSD300
from multibox import MultiBoxLoss
from priorbox import PriorBox

from dataset import COCODataset
from config import config

class SSD_simple(pl.LightningModule):

    def __init__(self,config : dict):
        super().__init__()
        self.config = config

        self.model = SSD300()

    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_nb):
        images, bboxes, labels = batch
        locs, confs = self(images)

        priors = PriorBox(self.config)
        priors = priors.create_priors().to(self.device)
        multibox_loss = MultiBoxLoss(num_classes=self.config['num_classes'],priors=priors,cfg=self.config,device=self.device)

        loc_loss, conf_loss = multibox_loss(locs, confs, bboxes, labels)
        loss = conf_loss + loc_loss

        logs = {'train_loss' : loss, 'conf_loss':conf_loss, 'loc_loss' : loc_loss}

        return {'loss' : loss, 'log' : logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        data = COCODataset('../val2017','../annotations/instances_val2017.json',split='TRAIN')
        train_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True,
                                               collate_fn=data.collate_fn, num_workers=4)
        return train_loader
    
    def val_dataloader(self):
        data = COCODataset('../val2017','../annotations/instances_val2017.json',split='TEST')
        val_loader = torch.utils.data.DataLoader(data, batch_size=4,
                                               collate_fn=data.collate_fn, num_workers=4)
        return val_loader
    
    def validation_step(self, batch, batch_nb):
        images, bboxes, labels = batch
        locs, confs = self(images)

        priors = PriorBox(self.config)
        priors = priors.create_priors().to(self.device)
        multibox_loss = MultiBoxLoss(num_classes=self.config['num_classes'],priors=priors,cfg=self.config,device=self.device)

        loc_loss, conf_loss = multibox_loss(locs, confs, bboxes, labels)
        loss = conf_loss + loc_loss
        return {'val_loss' : loss, 'conf_loss':conf_loss, 'loc_loss' : loc_loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


model = SSD_simple(config=config)

from pytorch_lightning import Trainer

trainer = Trainer(gpus=1)

trainer.fit(model)