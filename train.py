import torch
from config import config
import os
from torch.utils.tensorboard import SummaryWriter
import time

from priorbox import PriorBox
from model import SSD300
from multibox import MultiBoxLoss

from dataset import COCODataset

# Set the device
device = config['device']

priors = PriorBox(config).create_priors().to(device)

# Set up model
model = SSD300()
model = model.to(device)

# Set up loss crit
criterion = MultiBoxLoss(priors,config)
criterion = criterion.to(device)

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

# Set up data train and val
data_train = COCODataset('../val2017','../annotations/instances_val2017.json',split='TRAIN')
train_loader = torch.utils.data.DataLoader(data_train, batch_size=config['batch_size'], shuffle=True,
                                               collate_fn=data_train.collate_fn, num_workers=config['num_workers'])

data_val = COCODataset('../val2017','../annotations/instances_val2017.json',split='TEST')
val_loader = torch.utils.data.DataLoader(data_val, batch_size=config['batch_size'], shuffle=False,
                                               collate_fn=data_val.collate_fn, num_workers=config['num_workers'])

# Create model save dir
if not os.path.exists('weights'):
    os.mkdir('weights')

from tqdm import tqdm

def train_one_epoch(model, criterion, optimizer, train_loader, writer, config, epoch_no, log_every = 50):
    """Train the model for one complete epoch"""
    
    model.train()
    loc_loss = 0
    conf_loss = 0
    total_loss = 0

    for i, data in tqdm(enumerate(train_loader)):
        images, bboxes, labels = data

        images = images.to(device)
        bboxes = [x.to(device) for x in bboxes]
        labels = [x.to(device) for x in labels]

        optimizer.zero_grad()

        locs, confs = model(images)

        loss_l, loss_c = criterion(locs, confs, bboxes, labels)

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        loss = config['alpha'] * loss_l + loss_c

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        writer.add_scalar('Train/loc_loss', loss_l.item(), epoch_no * len(train_loader) + i)
        writer.add_scalar('Train/conf_loss', loss_c.item(), epoch_no * len(train_loader) + i)
        writer.add_scalar('Train/total_loss', loss.item(), epoch_no * len(train_loader) + i)

        if i % log_every == 0:
            # print stats
            stats = ' {}/{} Epochs | {}/{} batch | conf_loss: {:.5f} | loc_loss: {:.5f} | loss: {:.5f} | lr: {}'.format(epoch_no,
                                                                                                       config['num_epochs'],
                                                                                                       i+1,
                                                                                                       len(train_loader),
                                                                                                       loss_c.item(),
                                                                                                       loss_l.item(),
                                                                                                       loss.item(),
                                                                                                       get_lr(optimizer) 
                                                                                                    )
            print(stats)
    
    # Take avg here
    conf_loss /= len(train_loader)
    loc_loss /= len(train_loader)
    total_loss /= len(train_loader)

    print('{} epoch ended | conf_loss: {:.5f} | loc_loss: {:.5f} | loss: {:.5f}'.format(epoch_no, conf_loss, loc_loss, total_loss))

    writer.add_scalar('Train/avg loc_loss', loc_loss, epoch_no)
    writer.add_scalar('Train/avg conf_loss', conf_loss, epoch_no)
    writer.add_scalar('Train/avg total_loss', total_loss, epoch_no)

def val_one_epoch(model, criterion, val_loader, writer, config, epoch_no, log_every = 50):
    """Validate the model for one complete epoch"""
    
    model.eval()
    loc_loss = 0
    conf_loss = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, bboxes, labels = data

            images = images.to(device)
            bboxes = [x.to(device) for x in bboxes]
            labels = [x.to(device) for x in labels]

            locs, confs = model(images)

            loss_l, loss_c = criterion(locs, confs, bboxes, labels)

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            loss = config['alpha'] * loss_l + loss_c

            total_loss += loss.item()

            if i % log_every == 0:
                # print stats
                stats = 'Validating | {}/{} batch | conf_loss: {:.5f} | loc_loss: {:.5f} | loss: {:.5f} | lr: {}'.format(
                                                                                                        i+1,
                                                                                                        len(val_loader),
                                                                                                        loss_c.item(),
                                                                                                        loss_l.item(),
                                                                                                        loss.item(),
                                                                                                        get_lr(optimizer) 
                                                                                                        )
                print(stats)
    
    # Take avg here
    conf_loss /= len(val_loader)
    loc_loss /= len(val_loader)
    total_loss /= len(val_loader)

    print('Validation ended | conf_loss: {:.5f} | loc_loss: {:.5f} | loss: {:.5f}'.format(epoch_no, conf_loss, loc_loss, total_loss))

    writer.add_scalar('Val/avg loc_loss', loc_loss, epoch_no)
    writer.add_scalar('Val/avg conf_loss', conf_loss, epoch_no)
    writer.add_scalar('Val/avg total_loss', total_loss, epoch_no)

    return total_loss

def adjust_learning_rate(optimizer, gamma, step, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():

    print('Starting to train')

    epoch_no = 0
    step = 0
    writer = SummaryWriter(comment='name={}'.format(config['name']))
    start_training_time = time.time()

    num_epochs = config['num_epochs']


    for epoch_no in range(num_epochs):

        epoch_start_time = time.time()
        # Train a epoch
        train_one_epoch(model, criterion, optimizer, train_loader, writer, config, epoch_no, config['log_every_train'])

        # do validation
        val_loss = val_one_epoch(model, criterion, val_loader, writer, config, epoch_no, config['log_every_val'])

        epoch_end_time = time.time()

        print('Epoch took {:.3f} s.'.format(epoch_end_time - epoch_start_time))
        print('--' * 30)

        writer.flush()

        if epoch_no in config['epochs_lr']:
            print('Reducing lr by factor of',config['gamma'])
            step += 1
            adjust_learning_rate(optimizer, config['gamma'], step, config['lr'])
        
        if config['save_model']:
            if epoch_no in config['save_model_epochs']:
                print('Saving model now...')
                torch.save(model.state_dict(),'./weights/COCO-resnet-{}-{}-val_loss-{:.3f}.pth'.format(epoch_no, config['name'], val_loss))
        
    
    print('Training ended after {:.3f} min'.format((time.time() - start_training_time)/60))

# Run the training main function
main()
