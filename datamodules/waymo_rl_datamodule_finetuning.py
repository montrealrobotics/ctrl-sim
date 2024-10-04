import pytorch_lightning as pl 
from datasets.rl_waymo.dataset_ctrl_sim_finetuning import RLWaymoDatasetCtRLSimFineTuning 
from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim
from torch_geometric.loader import DataLoader

class RLWaymoDataModuleFineTuning(pl.LightningDataModule):

    def __init__(self,
                 cfg):
        super(RLWaymoDataModuleFineTuning, self).__init__()
        self.cfg = cfg
        self.cfg_dataset = cfg.dataset.waymo
        

    def setup(self, stage):
        self.train_dataset = RLWaymoDatasetCtRLSimFineTuning(self.cfg)
        self.val_dataset = RLWaymoDatasetCtRLSim(self.cfg, split_name='val') 


    def sample_real_indices(self):
        # Re-sample real dataset indices at the start of each training epoch
        self.train_dataset.sample_real_indices()


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.cfg.datamodule.train_batch_size, 
                          shuffle=True, 
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory,
                          drop_last=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.datamodule.val_batch_size,
                          shuffle=False,
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory,
                          drop_last=True)
