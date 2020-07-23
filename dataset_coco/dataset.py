from torch.utils.data import Dataset
import torchvision.datasets as dataset
import torch
import numpy as np
from dataset_coco.utils import transform

class COCODataset(Dataset):
    
    def __init__(self, img_path, ann_path, split):
        self.coco = dataset.CocoDetection(root = img_path, annFile = ann_path)
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}


    def __len__(self):
        return len(self.coco)
        # return 128
    
    def __getitem__(self,index):
        
        img, label = self.coco[index]
        # TODO check if he image is really image or just an image path
        bboxs = [x['bbox'] for x in label]

        if len(bboxs) == 0:
            bboxs = np.zeros((0,4))
        else:
            # bbox are stored as x,y,w,h
            bboxs = np.array(bboxs).reshape(-1,4)

        # convert to x1,y1,x2,y2 coordinate form also called as boundary form
        # this is done because the transforms are made in accordance to the boundary
        # coordinate
        for bbox in bboxs:
            bbox[2:] += bbox[:2]

        bboxs = torch.from_numpy(bboxs).float()


        bbox_labels = [x['category_id']-1 for x in label]

        if len(bbox_labels) == 0:
            bbox_labels = torch.zeros((0,91), dtype=torch.long)
        else:
            bbox_labels = torch.LongTensor(bbox_labels)

        # Applying augmentations
        image, bboxs, labels = transform(img,bboxs,bbox_labels,split=self.split)

        return image, bboxs, labels
    
    def collate_fn(self, batch):
        
        images = []
        boxes = []
        labels = []
        
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        
        images = torch.stack(images,dim = 0)
        return images, boxes, labels
