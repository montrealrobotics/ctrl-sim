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

def match_md_to_nocturne(md_sdc_pos, md_adv_pos, nocturne_dict):
    matched = not(md_sdc_pos[0] == 0 or md_adv_pos[0] == 0)
    nocturne_sdc_id = None 
    nocturne_adversary_id = None

    nocturne_interactive_ids = nocturne_dict['interactive_ids']
    veh_pos_0 = np.array([nocturne_dict['objects'][nocturne_interactive_ids[0]]['position'][0]['x'], nocturne_dict['objects'][nocturne_interactive_ids[0]]['position'][0]['y']])
    veh_pos_1 = np.array([nocturne_dict['objects'][nocturne_interactive_ids[1]]['position'][0]['x'], nocturne_dict['objects'][nocturne_interactive_ids[1]]['position'][0]['y']])

    if matched:
        dist_0_to_md_sdc = np.linalg.norm(veh_pos_0 - md_sdc_pos)
        dist_1_to_md_sdc = np.linalg.norm(veh_pos_1 - md_sdc_pos)

        if dist_0_to_md_sdc < dist_1_to_md_sdc:
            nocturne_sdc_id = nocturne_interactive_ids[0]
            nocturne_adversary_id = nocturne_interactive_ids[1]
            nocturne_sdc_pos = veh_pos_0 
            nocturne_adversary_pos = veh_pos_1
        else:
            nocturne_sdc_id = nocturne_interactive_ids[1]
            nocturne_adversary_id = nocturne_interactive_ids[0]
            nocturne_sdc_pos = veh_pos_1
            nocturne_adversary_pos = veh_pos_0

        if not (np.linalg.norm(md_sdc_pos - nocturne_sdc_pos) < 0.01 and np.linalg.norm(md_adv_pos - nocturne_adversary_pos) < 0.01):
            matched=False
    
    return nocturne_sdc_id, nocturne_adversary_id, matched


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    test_filenames = os.listdir(cfg.nocturne_waymo_val_interactive_folder)
    test_filenames = [file for file in test_filenames if 'tfrecord' in file]
    test_filenames = sorted(test_filenames)

    seed = 2024
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch.
    torch.cuda.manual_seed(seed)  # PyTorch, for CUDA.

    # This ensures that the test set is randomized
    file_ids = list(np.arange(len(test_filenames)))
    random.shuffle(file_ids)

    md_file_path = cfg.cat.md_scenarios
    md_test_filenames = os.listdir(md_file_path)
    md_test_filenames_id = [name[36:-21] for name in md_test_filenames]
    
    if not os.path.exists(cfg.cat.valid_md_scenarios):
        os.makedirs(cfg.cat.valid_md_scenarios, exist_ok=True)
    output_dict = {}

    count = 0
    for file_id in tqdm(file_ids):
        file = test_filenames[file_id]
        print(file[:-5], md_test_filenames_id[0])
        exit()
        if file[:-5] not in md_test_filenames_id:
            continue   
            
        md_idx = md_test_filenames_id.index(file[:-5])
        md_path = os.path.join(md_file_path, md_test_filenames[md_idx])
        # filtering consistent with cat (select_cases.py)
        with open(md_path, 'rb') as f:
            md_dict = pickle.load(f)
            
            track_len = md_dict['metadata']['track_length']
            # exclude partial tracks
            if track_len != 91:
                continue
            
            interactive_ids = md_dict['metadata']['objects_of_interest']
            md_sdc_id = md_dict['metadata']['sdc_id']
            
            # exclude scenarios where sdc is not an interacting agent
            if md_sdc_id not in interactive_ids:
                continue

            # other interacting agent is the adversary. Note that interactive split is 2 agent interaction scenarios
            interactive_ids.remove(md_sdc_id)
            md_adversary_id = interactive_ids[0]
        
        nocturne_path = os.path.join(cfg.nocturne_waymo_val_interactive_folder, file)
        with open(nocturne_path, 'r') as f:
            nocturne_dict = json.load(f)
            nocturne_sdc_id, nocturne_adversary_id, matched = match_md_to_nocturne(md_dict['tracks'][md_sdc_id]['state']['position'][0, :2], 
                                                                                   md_dict['tracks'][md_adversary_id]['state']['position'][0, :2],
                                                                                   nocturne_dict)
            if not matched:
                continue

        # rename md scenario files and copy into directory of valid scenarios
        new_md_path = os.path.join(cfg.cat.valid_md_scenarios, f'{count}.pkl')
        os.system(f'cp {md_path} {new_md_path}')
        
        output_dict[count] = {
            'md_path': new_md_path,
            'nocturne_path': nocturne_path,
            'md_sdc_id': md_sdc_id,
            'md_adversary_id': md_adversary_id,
            'nocturne_sdc_id': nocturne_sdc_id,
            'nocturne_adversary_id': nocturne_adversary_id
        }
        count += 1
    
    with open(cfg.cat.dict_path, 'wb') as f:
        pickle.dump(output_dict, f)
    print("Number of valid scenarios:", count)

main()