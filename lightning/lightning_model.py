from models.resnet50_backbone import SSD300
from models.multibox import MultiBoxLoss
import pytorch_lightning as pl
import torch
from dataset import COCODataset
from config import config

class SSD300_COCO(pl.LightningModule):

    def __init__(self,cfg):
        super(SSD300_COCO, self).__init__()
        self.cfg = cfg
        self.feature_extractor = SSD300()
        self.criterion = MultiBoxLoss(cfg=config)

    def forward(self, images):
        """images has dim as [*,3,300,300]"""
        locs, confs = self.feature_extractor(images)
        return locs, confs

    def prepare_data(self):
        self.train_data = COCODataset(self.cfg.train_data_path, self.cfg.train_annotate_path, 'TRAIN')
        self.val_data = COCODataset(self.cfg.val_data_path, self.cfg.val_annotate_path, 'TEST')

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            collate_fn=self.train_data.collate_fn
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            collate_fn=self.val_data.collate_fn,
        )
        return valid_loader

    def configure_optimizers(self):

        # TODO: Apply Tencent trick also
        optimizer = torch.optim.SGD(
            params=self.feature_extractor.parameters(),
            lr=self.cfg.train.lr,
            momentum=self.cfg.train.momentum,
            weight_decay=self.cfg.train.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.train.step_size,
            gamma=self.cfg.train.gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, bboxes, bbox_labels = batch
        locs, confs = self(images)

        # calc loss
        loc_loss, conf_loss = self.criterion.forward(locs, confs, bboxes, bbox_labels)
        loss = conf_loss + self.cfg.train.alpha * loc_loss

        loss_dict = {'conf_loss':conf_loss, 'loc_loss':loc_loss}
        return {'loss':loss, 'logs':loss_dict, 'progress_bar':loss_dict}
