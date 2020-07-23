import numpy as np
from models.box_utils import jaccard
import torch
import numpy as np
def mapCalc(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes):
    '''
    input:
    pred_boxes = list of length equalt to image in val dataset each image contains torch tensor[Nx4]
    pred_scores = list of length equalt to image in val dataset each image contains torch tensor[N]
    pred_labels = list of length equalt to image in val dataset each image contains torch tensor[N]
    gt_boxes = list of length equalt to image in val dataset each image contains torch tensor[N'x4]
    gt_labels = list of length equalt to image in val dataset each image contains torch tensor[N']
    num_classes=91
    output:
    numpy array of shape [Num_classes,1] containing map values

    '''
    # Asserting correctness
    assert (len(pred_boxes)==len(pred_labels)==len(pred_scores)==len(gt_boxes)==len(gt_labels))
    # defining per class
    pr_data_collector=[np.zeros((0,2)) for i in range (0,num_classes)]
    totalObjects=[0 for i in range (0,num_classes)]
    prCurves=[0 for i in range (0,num_classes)]
    aps=[0 for i in range (0,num_classes)]
    import tqdm
    for c in range(1,num_classes):
        print(f"At class : {c}",end ="\r")
        
        # don't  calculate map for backgroung 0= bg, 1-90 = foreground objects

        # TODO maybe use a smarter version for faster implementations 

        num_images=len(pred_boxes)
        img=0
        while(img<num_images):
            # import pdb;pdb.set_trace()
            pred_box=pred_boxes[img][torch.where(pred_labels[img]==c)].reshape((-1,4))
            pred_label=pred_labels[img][torch.where(pred_labels[img]==c)]
            pred_score=pred_scores[img][torch.where(pred_labels[img]==c)]

            gt_box=gt_boxes[img][torch.where(gt_labels[img]==c)].reshape((-1,4))
            gt_label=gt_labels[img][torch.where(gt_labels[img]==c)]

            # incrementing total object
            totalObjects[c]+=gt_box.size(0)

            # calculating overlaps
            # print(pred_box.shape,gt_box.shape,end="\r")
            overlaps=jaccard(pred_box,gt_box)
            # print(pred_box.shape,gt_box.shape,overlaps.shape)

            # initialisiing yes no list
            yesNoListcur=np.zeros((pred_box.size(0),2))

            # adding confidences
            yesNoListcur[...,1]=pred_score.cpu().numpy()

            # adding 0 for a fp and 1 for a tp
            if(gt_box.size(0)>0 and pred_box.size(0)>0):
                yesNoListcur[:,0][torch.where(overlaps.max(dim=1)[0].cpu()>=0.5)]=1

            # adding it to pr data collector
            pr_data_collector[c]=np.concatenate((pr_data_collector[c],yesNoListcur), axis=0)
        
            img+=1


        prCurves[c]=givePRCurve(pr_data_collector[c],totalObjects[c])

        propose = prCurves[c][:, 0]
        recall = prCurves[c][:, 1]
        
        # TODO optional logging in tensorboard
        # for i in range(len(recall)):
        #     # writer.add_scalars("prCurve",{f'iou_thresh={iou_thresh}':propose[i]},recall[i]*1000)
        # writer.flush()


        aps[c]=voc_ap(recall,propose)

    # for i,ap in enumerate(aps):
    #     print(i,ap) 
    
    return aps


            
def givePRCurve(pr_data_collector,total_num_objects):
    # pr_data_collector= nx2 (yes/no,conf)
    #returns nx3 (prec,recall,conf)
    #---------------------------------------------
    # sort the data according to scores
    pr_data_collector.view("f8,f8").sort(order=["f1"],axis=0)
    pr_data_collector=pr_data_collector[::-1,...]
    # print(pr_data_collector.shape)

    # now looping over it 
    tp=0
    fp=0
    my_pr_curve=np.zeros((pr_data_collector.shape[0],3))
    if(total_num_objects==0):
        return my_pr_curve
    
    for i, prPoint in enumerate(my_pr_curve):
        if(pr_data_collector[i][0]==1):
            tp+=1
        elif(pr_data_collector[i][0]==0):
            fp+=1
        else:
            input("An error occured")
        
        prPoint[0]=(tp/(tp+fp))# this is precision
        prPoint[1]=tp/total_num_objects
        prPoint[2]=pr_data_collector[i][1]

    return my_pr_curve

def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap