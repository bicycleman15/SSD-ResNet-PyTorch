from models.resnet50_backbone import SSD300
from models.multibox import MultiBoxLoss
import pytorch_lightning as pl
import torch
from dataset_coco.dataset import COCODataset
from models.box_utils import decode
from models.detect_utils import filter_boxes_batched
import torch.nn.functional as F
from models.map_utils import mapCalc
import os


class SSD300_COCO(pl.LightningModule):

    def __init__(self, cfg):
        super(SSD300_COCO, self).__init__()
        self.cfg = cfg
        self.hparams = cfg
        self.feature_extractor = SSD300()
        self.criterion = MultiBoxLoss(cfg=cfg)

    def forward(self, images):
        """images has dim as [*,3,300,300]"""
        locs, confs = self.feature_extractor(images)
        return locs, confs

    def prepare_data(self):  # TODO : fix here train
        self.train_data = COCODataset(self.cfg.data.train_data_path, self.cfg.data.train_annotate_path, 'TRAIN')
        self.val_data = COCODataset(self.cfg.data.val_data_path, self.cfg.data.val_annotate_path, 'TEST')

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=True,
            collate_fn=self.train_data.collate_fn
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
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
        loc_loss, conf_loss = self.criterion(locs, confs, bboxes, bbox_labels)
        loss = conf_loss + self.cfg.train.alpha * loc_loss

        prog_dict = {'conf_l': conf_loss, 'loc_l': loc_loss}
        log_dict = {'conf_l': conf_loss, 'loc_l': loc_loss, 'train_loss': loss}
        return {'loss': loss, 'log': log_dict, 'progress_bar': prog_dict}

    def validation_step(self, batch, batch_idx):
        images, bboxes, bbox_labels = batch
        locs, confs = self(images)

        # calc loss
        loc_loss, conf_loss = self.criterion(locs, confs, bboxes, bbox_labels)
        loss = conf_loss + self.cfg.train.alpha * loc_loss

        for i in range(locs.size(0)):
            # Decode targets here if possible
            locs[i] = decode(locs[i], self.criterion.priors, self.cfg.priors.variance)

        confs = F.softmax(confs, dim=2)
        scores, idxs = confs.max(dim=2)

        triplet_of_preds = filter_boxes_batched(locs, scores, idxs)
        # triplet of preds contains (predbboxes, predscores, predlabels)
        # where predbboxes is a list of size locs.size(0) or batch_size
        # each item of list is a tensor of filtered bboxes tensor in xy coords

        # predictions and gts will be used in validation end function to calculate APs

        return {'loc_loss': loc_loss, 'conf_loss': conf_loss, 'loss': loss, 'predictions' : triplet_of_preds, 'gt' : (bboxes, bbox_labels)}

    def validation_epoch_end(self, outputs):
        # TODO : also collect decoded boxes here to calc mAP and stuff (done)
        # Think of a cleaner way to take average
        conf_loss = sum([x['conf_loss'] for x in outputs])
        loc_loss = sum([x['loc_loss'] for x in outputs])
        loss = sum([x['loss'] for x in outputs])

        conf_loss /= len(outputs)
        loc_loss /= len(outputs)
        loss /= len(outputs)

        pred_boxes = []
        pred_scores = []
        pred_labels = []
        gt_boxes = []
        gt_labels = []

        # Now collect all the item in respective lists
        for x in outputs:
            pred_boxes.extend(x['predictions'][0])
            pred_scores.extend(x['predictions'][1])
            pred_labels.extend(x['predictions'][2])

            gt_boxes.extend(x['gt'][0])
            gt_labels.extend(x['gt'][1])

        # incorporate background class by adding +1
        gt_labels = [x+1 for x in gt_labels]
        
        aps = mapCalc(pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,self.cfg.basic.num_classes)
        
        c=1
        text=""
        while( c < self.cfg.basic.num_classes):
            self.logger.experiment.add_scalar(f"APs/Class :{c}", aps[c],self.current_epoch)
            text+=('Class : {} | AP : {} \n'.format(c,aps[c]))
            c+=1
        # Add in text form
 
        self.logger.experiment.add_text("AP Summary", text ,self.current_epoch)
        


        # Save model right here now, to save space
        # torch.save(self.feature_extractor.state_dict(), '{}/resnet-ssd-coco-{}'.format(self.cfg.train.model_save_path,loss))
        # print(pred_boxes[0].shape, gt_boxes[0].shape)
        # dont return here anything, instead do your own eval
        return {'val_loss': loss, 'log': {'val_loss': loss}, 'progress_bar': {'avg_val_loss': loss}}
