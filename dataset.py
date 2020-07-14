import torch
from torch.utils.data import Dataset
import torchvision.datasets as dataset
import torchvision.transforms.functional as FT
import numpy as np
from utils import transform

class COCODataset(Dataset):
    
    def __init__(self, img_path, ann_path, split):
        self.coco = dataset.CocoDetection(root = img_path, annFile = ann_path)
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}


    def __len__(self):
        return len(self.coco)
    
    def __getitem__(self,index):
        
        img, label = self.coco[index]

        # TODO check if he image is really image or just an image path

        
        bboxs = [x['bbox'] for x in label]

        # bbox are stored as x,y,w,h
        bboxs = np.array(bboxs)

        # convert to x1,y1,x2,y2 coordinate form aslo called as boundary form
        # this is done because the transforms are made in accordance to the boundary
        # coordinate
        for bbox in bboxs:
            bbox[2:] += bbox[:2]

        bboxs = torch.from_numpy(bboxs).float()


        bbox_labels = [x['category_id']-1 for x in label]
        bbox_labels = torch.LongTensor(bbox_labels)

        # Applying augmentations
        image, bboxs, labels = transform(img,bboxs,bbox_labels,split=self.split)
        # bboxs = [bbox.float() for bbox in bboxs]
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