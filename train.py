from lightning.lightning_model import SSD300_COCO
from omegaconf import OmegaConf
import pytorch_lightning as pl

from lightning.utils import set_seed
import argparse

# TODO :- remove config and replace with cfg yaml file (done)
# TODO :- make train working with scheduler and everything, logs too (done)
# TODO :- see saving model to take up less space (done)
# TODO :- make inference.py file/functions so that can work with it in validation_end() func, and calc AP (remaining)

if __name__ == '__main__':
    # Set seed
    set_seed()

    # parse config
    config = OmegaConf.load('config.yaml')
    print(config.pretty())

    print('Loading Model....')
    model = SSD300_COCO(cfg=config)

    # TODO : add gpus
    # parser = argparse.ArgumentParser()
    # parser.add_argument('gpus', default=1)

    # Create model save checkpoint
    save_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=5,
        save_weights_only=True
    )
    lr_logger = pl.callbacks.LearningRateLogger()

    trainer = pl.Trainer(
        checkpoint_callback=save_checkpoint,
        callbacks=[lr_logger],
        gpus=1
    )

    trainer.fit(model)

