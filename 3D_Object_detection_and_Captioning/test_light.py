import os
import torch
import hydra
from importlib import import_module
import pytorch_lightning as pl
from minsu3d.data.joint_data_module import JointDataModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    # fix the seed
    pl.seed_everything(cfg.global_test_seed, workers=True)

    print("=> initializing trainer...")
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)

    output_path = os.path.join(cfg.exp_output_root_path, "inference", cfg.model.inference.split, "predictions")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = JointDataModule(cfg)

    print("=> initializing model...")
    model = getattr(import_module("minsu3d.model"), cfg.model.network.module)(cfg)

    if cfg.model.inference.split == "test":
        # For hidden test set without GT, turn off evaluation
        cfg.model.inference.evaluate = False

    print("=> start inference...")
    
    checkpoint = torch.load(cfg.model.ckpt_path)
    trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]  # TODO
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == '__main__':
    main()


