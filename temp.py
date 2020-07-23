from lightning.lightning_model import SSD300_COCO
from omegaconf import OmegaConf
import pytorch_lightning as pl
from dataset_coco.dataset import COCODataset
from models.detect_utils import filter_boxes_batched
from models.box_utils import decode
import torch.nn.functional as F

from lightning.utils import set_seed

if __name__ == '__main__':
    # Set seed
    set_seed()

    # parse config
    config = OmegaConf.load('config.yaml')
    print(config.pretty())

    print('Loading Model....')
    model = SSD300_COCO(cfg=config)
    model.eval()

    data = COCODataset(config.data.val_data_path, config.data.val_annotate_path, 'TEST')
    print('data loaded')
    print(len(data))
    for x in data:
        image, bboxes, labels = x

        locs, confs = model(image.unsqueeze(0))

        for i in range(locs.size(0)):
            locs[i] = decode(locs[i] , model.criterion.priors, config.priors.variance)

        confs = F.softmax(confs, dim=2)
        scores, idxs = confs.max(dim=2)
        filtered_bboxes, filtered_confs, filtered_labels = filter_boxes_batched(locs, scores, idxs)
        #
        print(filtered_bboxes)
        print(filtered_confs)
        print(filtered_labels)
        # print(len(filtered_labels))
        break
