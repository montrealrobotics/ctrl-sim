import json
import os
import sys
import time
import argparse
import pickle
import random
import shutil

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import nocturne
import pdb
from tqdm import tqdm
from scipy.spatial import distance
from cfgs.config import CONFIG_PATH

# we need to split the val set into a val set and a test set (where we select 2500 scenes for the test set)
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    test_filenames = os.listdir(cfg.nocturne_waymo_val_folder)
    test_filenames = [file for file in test_filenames if 'tfrecord' in file]
    test_filenames = sorted(test_filenames)

    seed = 2024
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch.
    torch.cuda.manual_seed(seed)  # PyTorch, for CUDA.

    file_ids = list(np.arange(len(test_filenames)))
    random.shuffle(file_ids)
    test_file_ids = file_ids[:2500] # test set contains 2500 scenes
    val_file_ids = file_ids[2500:]

    test_files = [test_filenames[file_id] for file_id in test_file_ids]
    val_files = [test_filenames[file_id] for file_id in val_file_ids]
    d = {'test_filenames': test_files}
    with open(os.path.join(cfg.dataset_root, 'test_filenames.pkl'), 'wb') as f:
        pickle.dump(d, f)
    
    preprocess_val = os.path.join(cfg.dataset_root, 'preprocess/val')
    preprocess_test = os.path.join(cfg.dataset_root, 'preprocess/test')
    if not os.path.exists(preprocess_test):
        os.makedirs(preprocess_test, exist_ok=True)
    preprocess_files_val = os.listdir(preprocess_val)

    for file in tqdm(test_files):
        # move preprocess files
        output_file = f"{file[:-5]}_physics.pkl"
        if output_file in preprocess_files_val:
            old_file_path = os.path.join(preprocess_val, output_file)
            new_file_path = os.path.join(preprocess_test, output_file)
            shutil.move(old_file_path, new_file_path)

    # 9701 2500 9671 2489
    print(len(val_files), len(test_files), len(os.listdir(preprocess_val)), len(os.listdir(preprocess_test))) 

main()