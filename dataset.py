import torch
from torch.utils.data import Dataset
import torchvision.datasets as dataset
import torchvision.transforms.functional as FT
import numpy as np

class COCODataset(Dataset):
    
    def __init__(self, img_path, ann_path):
        self.coco = dataset.CocoDetection(root = img_path, annFile = ann_path)
    
    def __len__(self):
        return len(self.coco)
    
    def __getitem__(self,index):
        
        img, label = self.coco[index]
        
        bboxs = [x['bbox'] for x in label]
        bboxs = np.array(bboxs)
        # convert to boundary coord
        bboxs[:,2:] += bboxs[:,:2]
        bboxs = torch.from_numpy(bboxs)

        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height])
        new_boxes = bboxs / old_dims

        bbox_labels = [x['category_id']-1 for x in label]
        bbox_labels = torch.LongTensor(bbox_labels)

        # TODO :- apply transforms below 
        new_img = FT.resize(img, (300,300))
        new_img = FT.to_tensor(new_img)
        
        return new_img, new_boxes,bbox_labels
    
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