import json
import os
import time
import hydra
import numpy as np
from tqdm import tqdm

from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim
from cfgs.config import CONFIG_PATH

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    cfg.dataset.waymo.preprocess = False
    dset = RLWaymoDatasetCtRLSim(cfg, split_name=cfg.preprocess_rl_waymo.mode)

    chunk = list(range(cfg.preprocess_rl_waymo.chunk_idx * cfg.preprocess_rl_waymo.chunk_size, (cfg.preprocess_rl_waymo.chunk_idx + 1) * cfg.preprocess_rl_waymo.chunk_size))
    if len(dset.files) < chunk[0]:
        raise ValueError("chunk_idx is too large for dataset size.")
    elif len(dset.files) < chunk[-1]:
        chunk = [c for c in chunk if c < len(dset)]

    for idx in tqdm(chunk):
        # search for file with at least 2 agents
        proceed = False 
        terminate = False
        while not proceed:
            with open(dset.files[idx], 'r') as file:
                data = json.load(file)
                if len(data['objects']) == 1:
                    idx += 1
                    if idx not in chunk:
                        terminate = True
                        break
                else:
                    proceed = True 
        
        if terminate:
            break
        
        dset.get_data(data, idx)
    
    print("Done!")

main()
