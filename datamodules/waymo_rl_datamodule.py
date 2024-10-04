import pytorch_lightning as pl 
from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim
from datasets.rl_waymo.dataset_ctg_plus_plus import RLWaymoDatasetCTGPlusPlus
from torch_geometric.loader import DataLoader
import os

# this ensures CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 

class RLWaymoDataModule(pl.LightningDataModule):

    def __init__(self,
                 cfg):
        super(RLWaymoDataModule, self).__init__()
        self.ctg_plus_plus = cfg.model.ctg_plus_plus
        self.cfg = cfg
        self.cfg_dataset = cfg.dataset.waymo
        

    def setup(self, stage):
        if self.ctg_plus_plus:
            self.train_dataset = RLWaymoDatasetCTGPlusPlus(self.cfg, split_name='train')
            self.val_dataset = RLWaymoDatasetCTGPlusPlus(self.cfg, split_name='val') 
        else:
            self.train_dataset = RLWaymoDatasetCtRLSim(self.cfg, split_name='train')
            self.val_dataset = RLWaymoDatasetCtRLSim(self.cfg, split_name='val') 


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.cfg.datamodule.train_batch_size, 
                          shuffle=True, 
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory,
                          drop_last=True,
                          worker_init_fn=worker_init_fn)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.datamodule.val_batch_size,
                          shuffle=False,
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory,
                          drop_last=True)