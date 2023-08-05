import os
import hydra
import pytorch_lightning as pl
from minsu3d.callback import *
from importlib import import_module
from minsu3d.data.joint_data_module import JointDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, gpu_cache_clean_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = JointDataModule(cfg)

    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=output_path)

    print("==> initializing monitor ...")
    callbacks = init_callbacks(cfg)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.model.trainer)

    print("==> initializing model ...")
    model = getattr(import_module("minsu3d.model"), cfg.model.network.module)(cfg)

    print("==> start training ...")
    if cfg.model.ckpt_path != None:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)
    else:
        trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
