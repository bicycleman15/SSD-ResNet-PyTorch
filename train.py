
from config import config

from dataset import COCODataset

import torch
from torch.utils.tensorboard import SummaryWriter
from train_utils import _train_model, _eval_model, _get_lr
import time
import torch.utils.data as data
import os
from multibox import MultiBoxLoss
from model import SSD300
from priorbox import PriorBox
from config import config

"""Performs training of a specified model.
    
Input params:
    config_file: Takes in configurations to train with 
"""

def train(config):
    """
    Function where actual training takes place
    Args:
        config (dict) : Configuration to train with
    """
    num_classes = config['num_classes'] # +1 for background class
    
    print('Starting to Train Model...')

    train_data = COCODataset('dataset/train2017','dataset/annotations/instances_train2017.json',split="train")
    train_loader = data.DataLoader(
        train_data, batch_size=config['batch_size'], num_workers=8, shuffle=True,collate_fn=train_data.collate_fn
    )

    val_data = COCODataset('dataset/val2017','dataset/annotations/instances_val2017.json',split='test')
    val_loader = data.DataLoader(
        val_data, batch_size=4, num_workers=8, collate_fn=val_data.collate_fn,shuffle=False
    )

    print('Initializing Model...')
    model = SSD300()
    if torch.cuda.is_available():
        model = model.cuda("cuda:2")

    print('Initializing Loss Method...')

    # TODO why are you taking this first priors already
    priors = PriorBox(cfg = config)
    priors = priors.create_priors()
    criterion = MultiBoxLoss(num_classes,priors,config)
    val_criterion = MultiBoxLoss(num_classes,priors,config)

    if torch.cuda.is_available():
        criterion = criterion.cuda("cuda:2")
        val_criterion = val_criterion.cuda("cuda:2")

    print('Setup the Optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    
    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    patience = config['patience']

    print('Starting Training')

    
    writer = SummaryWriter(comment='lr={} task={}'.format(config['lr'], config['name']))
    t_start_training = time.time()
    iterations=0
    for epoch in range(starting_epoch, num_epochs):
        totalTrainLoss=0
        maxtloss=0
        for batch_no,curdata in enumerate(train_loader):
            print("[Epoch: {0} / {1} | Batch : {2} / {3} ]|".format(
                      epoch + 1,
                      num_epochs,
                      batch_no,
                      len(train_loader)
                    ),end="\r")
            (imgs, bboxs, labels)=curdata
            optimizer.zero_grad()

            imgs = imgs.cuda("cuda:2")
            locs, confs = model(imgs)

            summed_multibox_loss, conf_loss,loc_loss = criterion.forward(locs, confs, bboxs, labels)
            totalTrainLoss+=summed_multibox_loss.item()
            maxtloss=max(maxtloss,summed_multibox_loss.item())
            summed_multibox_loss.backward()
            optimizer.step()

            if(batch_no%500==0):
                writer.add_scalar("Train Loss vs Iteration", summed_multibox_loss.item(), iterations)
                print("[Epoch: {0} / {1} | Batch : {2} / {3} ]| Batch Loss : {4:.4}".format(
                      epoch + 1,
                      num_epochs,
                      batch_no,
                      len(train_loader),
                      summed_multibox_loss
                    )
                )
            iterations+=1

        print("#"*50,"\n","Epoch {} has completed.\nTotal Train Loss : {} | maxi at batch : {}".format(epoch,totalTrainLoss,maxtloss))
        totalValLoss=evaluate_val(val_loader,model,criterion)
        writer.add_scalar("Train Loss vs Epoch", totalTrainLoss, epoch)
        writer.add_scalar("Train Loss vs Epoch", totalValLoss, epoch)
        print("\nTotal Val Loss : {} ".format(totalValLoss),"\n")
        writer.add_scalar("LR vs Epoch", _get_lr(optimizer), epoch)
        scheduler.step(totalValLoss)
        writer.flush()

        #Saving the model
        if bool(config['save_model']):
            file_name = 'model_{}_epoch_{}.pth'.format(config['name'], epoch+1)
            torch.save({
                'model_state_dict': model.state_dict()
            }, './weights/{}'.format(file_name))

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    writer.flush()
    writer.close()

def evaluate_val(dataloader,model,criterion):
    totalLoss=0
    with torch.no_grad():
        for batch_no,(imgs, bboxs, labels) in enumerate(dataloader):
            print("Evaluation on val dataset going on Batch : {0} / {1} ]|".format(
                      batch_no,
                      len(dataloader)
                    ),end="\r")


            imgs = imgs.cuda("cuda:2")
            locs, confs = model(imgs)

            summed_multibox_loss, conf_loss,loc_loss = criterion.forward(locs, confs, bboxs, labels)
            totalLoss+=summed_multibox_loss.item()
    return totalLoss

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 
if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
