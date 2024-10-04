import os 
import hydra
from models.ctrl_sim import CtRLSim
from models.ctg_plus_plus import CTGPlusPlus
from datamodules.waymo_rl_datamodule import RLWaymoDataModule 
from datamodules.waymo_rl_datamodule_finetuning import RLWaymoDataModuleFineTuning

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from cfgs.config import CONFIG_PATH

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    pl.seed_everything(cfg.train.seed, workers=True)

    # checkpoints saved here
    save_dir = os.path.join(cfg.train.save_dir, cfg.train.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if cfg.train.finetuning:
        datamodule = RLWaymoDataModuleFineTuning(cfg)
    else:
        datamodule = RLWaymoDataModule(cfg)
    
    if cfg.train.finetuning:
        model = CtRLSim.load_from_checkpoint(os.path.join(save_dir, 'model.ckpt'), cfg=cfg, data_module=datamodule)
    
    elif cfg.model.ctg_plus_plus:
        model = CTGPlusPlus(cfg)
        monitor = 'state_mse'
    else:
        model = CtRLSim(cfg)
        monitor = 'val_loss'
    
    if cfg.train.finetuning:
        # we save checkpoint at every epoch to support resume training, but we take the last epoch checkpoint for evaluation
        model_checkpoint = ModelCheckpoint(monitor=None, every_n_epochs=1, save_last=True, dirpath=save_dir, filename='model_finetuning')
    else: 
        # we always track the best epoch checkpoint for evaluation or resume training.   
        model_checkpoint = ModelCheckpoint(monitor=monitor, save_top_k=1, save_last=True, mode='min', dirpath=save_dir, filename='model')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=-1)
    wandb_logger = WandbLogger(
        project=cfg.train.wandb_project,
        name=cfg.train.run_name,
        entity=cfg.train.wandb_entity,
        log_model=False,
        save_dir=save_dir
    )
    if cfg.train.track:
        logger = wandb_logger 
    else:
        logger = None
    
    # resume training
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    if not cfg.train.finetuning:
        for file in files_in_save_dir:
            if file.endswith('.ckpt') and 'last' in file:
                ckpt_path = os.path.join(save_dir, file)
                print("Resuming from checkpoint: ", ckpt_path)
    
    trainer = pl.Trainer(accelerator=cfg.train.accelerator,
                         devices=cfg.train.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary, model_checkpoint, lr_monitor],
                         max_steps=cfg.train.max_steps,
                         check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                         precision=cfg.train.precision,
                         limit_train_batches=cfg.train.limit_train_batches, # train on smaller dataset
                         limit_val_batches=cfg.train.limit_val_batches,
                         gradient_clip_val=cfg.train.gradient_clip_val,
                         logger=logger
                        )
    
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    if cfg.train.finetuning:
        # NOTE: A hacky way to get around save_hyperparameters() raising error 
        # when specifying cfg=cfg, data_module=datamodule in load_from_checkpoint
        old_ckpt_file = torch.load(os.path.join(save_dir, 'model.ckpt'))
        ckpt_file = torch.load(os.path.join(save_dir, 'model_finetuning.ckpt'))
        ckpt_file['hyper_parameters'] = old_ckpt_file['hyper_parameters']
        # overwrite old model.ckpt file
        torch.save(ckpt_file, os.path.join(save_dir, 'model.ckpt'))
        os.remove(os.path.join(save_dir, 'model_finetuning.ckpt'))

if __name__ == '__main__':
    main()


