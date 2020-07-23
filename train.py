from lightning.lightning_model import SSD300_COCO
from omegaconf import OmegaConf
import pytorch_lightning as pl
from lightning.utils import set_seed
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
# TODO : add gpus
parser = argparse.ArgumentParser(description='Lightning SSD Training')
parser.add_argument('--gpus', default='0', 
    help='provide gpus in comma separeated fashion eg for gpus 1,2 provide --gpus 1,2')
args = parser.parse_args()

# asserting corect input for gpus
gpu_list=args.gpus.split(",")
assert len(gpu_list)== sum(list(map(lambda x: x.isnumeric(),gpu_list)))

# creating a numeric gpu list
gpu_list=list(map(int ,gpu_list))

# TODO :- make inference.py file/functions so that can work with it in validation_end() func, and calc AP (remaining)

if __name__ == '__main__':
    # Set seed
    set_seed()

    # parse config
    config = OmegaConf.load('config.yaml')
    print(config.pretty())

    print('Loading Model....')
    model = SSD300_COCO(cfg=config)

    
    logger=TensorBoardLogger(save_dir="runs")
    # Create model save checkpoint
    save_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=5,
        save_weights_only=True
    )
    lr_logger = pl.callbacks.LearningRateLogger()

    # Finding gpu configuration

    if(len(gpu_list)>1):
        # Use multi gpu training

        trainer = pl.Trainer(
            checkpoint_callback=save_checkpoint,
            callbacks=[lr_logger],
            gpus=gpu_list,
            train_percent_check=0.01,
            val_percent_check=0.01,
            distributed_backend='ddp',
            logger=logger
        )
    else:
        trainer = pl.Trainer(
            checkpoint_callback=save_checkpoint,
            callbacks=[lr_logger],
            gpus=gpu_list,
            train_percent_check=0.01,
            val_percent_check=0.01,
            logger=logger
        )


    trainer.fit(model)

