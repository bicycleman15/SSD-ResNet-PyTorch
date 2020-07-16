from dataset import CTData
from model import CovidCT
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

def train(config : dict):
    """
    Function where actual training takes place
    Args:
        config (dict) : Configuration to train with
    """
    num_classes = config['num_classes'] # +1 for background class
    
    print('Starting to Train Model...')

    train_data = COCODataset('../train2017','../annotations/instances_train2017.json',split="train")
    train_loader = data.DataLoader(
        train_data, batch_size=config['batch_size'], num_workers=8, shuffle=True
    )

    val_data = CTData('../val2017','../annotations/instances_val2017.json',split='test')
    val_loader = data.DataLoader(
        val_data, batch_size=4, num_workers=8, shuffle=False
    )

    print('Initializing Model...')
    model = SSD300()
    if torch.cuda.is_available():
        model = model.cuda()

    print('Initializing Loss Method...')

    # TODO why are you taking this first priors already
    priors = PriorBox(cfg = config)
    criterion = MultiBoxLoss(num_classes,priors,config)
    val_criterion = MultiBoxLoss(num_classes,priors,config)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    print('Setup the Optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    
    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    patience = config['patience']
    log_train = int(config['log_train'])
    log_val = int(config['log_val'])

    best_val_auc = float(0)

    print('Starting Training')

    writer = SummaryWriter(comment='lr={} task={}'.format(config['lr'], config['name']))
    t_start_training = time.time()

    for epoch in range(starting_epoch, num_epochs):

        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()  # timer for entire epoch

        conf_matrix, train_loss, train_auc, train_acc = _train_model(
            model, train_loader, epoch, num_epochs, config['batch_size'], optimizer, criterion, writer, current_lr, log_train)
        
        print("Confusion matrix Train here...")
        print(conf_matrix)

        conf_matrix_val, val_loss, val_auc, val_acc = _eval_model(
            model, val_loader, epoch, num_epochs, config['batch_size'], val_criterion, writer, log_val)

        print("Confusion matrix Val here...")
        print(conf_matrix_val)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time

        print("train loss : {:0.4f} | train auc {:0.4f} | train acc {:0.4f} | val loss {:0.4f} | val auc {:0.4f} | val acc {:0.4f} |  elapsed time {} s".format(
            train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, delta))

        print('-' * 30)

        writer.flush()

        if val_acc > best_val_auc:
            best_val_auc = val_acc

        # Decide when to save model
            if bool(config['save_model']):
                file_name = 'model_{}_val_acc_{:0.4f}_epoch_{}.pth'.format(config['name'], val_acc, epoch+1)
                torch.save({
                    'model_state_dict': model.state_dict()
                }, './weights/{}'.format(file_name))

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    writer.flush()
    writer.close()

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
