import os
import sys
import json
import glob
import hydra
import torch_scatter
import torch
import pickle
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
torch.set_printoptions(threshold=100000)
import numpy as np
np.set_printoptions(suppress=True)

from utils.data import *
from utils.geometry import apply_se2_transform, angle_sub_tensor

from datasets.rl_waymo.dataset import RLWaymoDataset
from cfgs.config import CONFIG_PATH

class RLWaymoDatasetCtRLSim(RLWaymoDataset):
    def __init__(self, cfg, split_name='train', mode='train'):
        super(RLWaymoDatasetCtRLSim, self).__init__(cfg, split_name, mode)

    
    def select_random_origin_agent(self, agent_states, moving_mask):
        # search for moving agent that exists at first timestep
        valid_idxs = np.where((agent_states[:, 0, -1] == 1) * moving_mask)[0]
        rand_idx = np.random.choice(len(valid_idxs))

        return valid_idxs[rand_idx]

    
    def get_data(self, data, idx):
        if self.preprocess:
            idx = data['idx']
            num_agents = data['num_agents']
            road_points = data['road_points']
            road_types = data['road_types']
            ag_data = data['ag_data']
            ag_actions = data['ag_actions']
            ag_types = data['ag_types']
            last_exist_timesteps = data['last_exist_timesteps']
            ag_rewards = data['ag_rewards']
            veh_edge_dist_rewards = data['veh_edge_dist_rewards']
            veh_veh_dist_rewards = data['veh_veh_dist_rewards']
            filtered_ag_ids = data['filtered_ag_ids']
            ag_goals = data['ag_goals']
            
        else:
            agent_data = data['objects']
            num_agents = len(agent_data)

            road_points, road_types, road_edge_polylines = self.get_roads(data)
            ag_data, ag_actions, ag_rewards, ag_types, ag_goals, parked_ids, incomplete_ids, last_exist_timesteps = self.extract_rawdata(agent_data)
            # zero out reward when timestep does not exist
            veh_edge_dist_rewards = self.compute_dist_to_nearest_road_edge_rewards(ag_data.copy(), road_edge_polylines) * ag_data[:, :, -1]
            veh_veh_dist_rewards = self.compute_dist_to_nearest_vehicle_rewards(ag_data.copy()) * ag_data[:, :, -1]

            raw_ag_ids = np.arange(num_agents).tolist()
            # no point in training on vehicles that don't have even one valid timestep
            filtered_ag_ids = list(filter(lambda x: x not in incomplete_ids, raw_ag_ids))
            assert len(filtered_ag_ids) > 0
            
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            to_pickle = dict()
            to_pickle['idx'] = idx
            to_pickle['num_agents'] = num_agents 
            to_pickle['road_points'] = road_points
            to_pickle['road_types'] = road_types
            to_pickle['ag_data'] = ag_data 
            to_pickle['ag_actions'] = ag_actions 
            to_pickle['ag_types'] = ag_types 
            to_pickle['last_exist_timesteps'] = last_exist_timesteps 
            to_pickle['veh_edge_dist_rewards'] = veh_edge_dist_rewards
            to_pickle['veh_veh_dist_rewards'] = veh_veh_dist_rewards 
            to_pickle['ag_rewards'] = ag_rewards
            to_pickle['filtered_ag_ids'] = filtered_ag_ids
            if self.cfg_dataset.preprocess_simulated_data:
                to_pickle['focal_agent_idx'] = data['focal_agent_idx']

            assert ag_goals is not None
            to_pickle['ag_goals'] = ag_goals
            
            with open(os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl'), 'wb') as f:
                pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        all_rewards = self.compute_rewards(ag_data, ag_rewards, veh_edge_dist_rewards, veh_veh_dist_rewards)
        rtgs = np.cumsum(all_rewards[:, ::-1], axis=1)[:, ::-1]
        
        if self.mode == 'eval':
            return rtgs, road_points, road_types

        rtgs = np.concatenate([rtgs[:, :, :1], rtgs[:, :, 3:5]], axis=2)
        rtgs[:, :, 0] = ((np.clip(rtgs[:, :, 0], a_min=self.cfg_dataset.min_rtg_pos, a_max=self.cfg_dataset.max_rtg_pos) - self.cfg_dataset.min_rtg_pos)
                             / (self.cfg_dataset.max_rtg_pos - self.cfg_dataset.min_rtg_pos))
        rtgs[:, :, 1] = ((np.clip(rtgs[:, :, 1], a_min=self.cfg_dataset.min_rtg_veh, a_max=self.cfg_dataset.max_rtg_veh) - self.cfg_dataset.min_rtg_veh)
                             / (self.cfg_dataset.max_rtg_veh - self.cfg_dataset.min_rtg_veh))
        rtgs[:, :, 2] = ((np.clip(rtgs[:, :, 2], a_min=self.cfg_dataset.min_rtg_road, a_max=self.cfg_dataset.max_rtg_road) - self.cfg_dataset.min_rtg_road)
                             / (self.cfg_dataset.max_rtg_road - self.cfg_dataset.min_rtg_road))
        
        goals = ag_goals
        moving_ids = np.where(np.linalg.norm(ag_data[:, 0, :2] - goals[:, 0, :2], axis=1) > self.cfg_dataset.moving_threshold)[0]
        goals = ag_goals[filtered_ag_ids, 0]
        
        # find max timestep to set as present timestep such that there exists an agent with train_context_length future timesteps
        max_timestep = np.max(last_exist_timesteps[moving_ids]) - (self.cfg_dataset.train_context_length - 1)
        # In this case, there will be some future timesteps such that all agents do not exist
        if max_timestep < 0:
            max_timestep = 0
        origin_t = np.random.randint(0, max_timestep+1)

        timesteps = np.arange(self.cfg_dataset.train_context_length) + origin_t
        timesteps = np.repeat(timesteps[np.newaxis, :, np.newaxis], self.cfg_dataset.max_num_agents, 0)
        agent_states = ag_data[filtered_ag_ids, origin_t:origin_t+self.cfg_dataset.train_context_length]
        agent_types = ag_types[filtered_ag_ids]
        actions = ag_actions[filtered_ag_ids, origin_t:origin_t+self.cfg_dataset.train_context_length]
        rtgs = rtgs[filtered_ag_ids, origin_t:origin_t+self.cfg_dataset.train_context_length]

        # filter for agents that move at least 0.05 metres
        moving_agent_mask = np.isin(filtered_ag_ids, moving_ids)
        # randomly choose moving agent to be at origin
        origin_agent_idx = self.select_random_origin_agent(agent_states, moving_agent_mask)

        agent_states, agent_types, actions, rtgs, goals, moving_agent_mask, new_origin_agent_idx = self.select_relevant_agents(agent_states, agent_types, actions, rtgs, goals, origin_agent_idx, 0, moving_agent_mask)
        actions = self.discretize_actions(actions)
        if not self.cfg_model.decision_transformer:
            rtgs = self.discretize_rtgs(rtgs)
        
        num_polylines = len(road_points)
        if num_polylines == 0:
            d = MotionData({})
            no_road_feats = True 
        else:
            agent_states, road_points, road_types, goals = self.normalize_scene(agent_states, road_points, road_types, goals, new_origin_agent_idx)
            d = dict()
            d['idx'] = idx
            # need to add batch dim as pytorch_geometric batches along first dimension of torch Tensors
            d['agent'] = from_numpy({
                'agent_states': add_batch_dim(agent_states),
                'agent_types': add_batch_dim(agent_types), 
                'goals': add_batch_dim(goals),
                'actions': add_batch_dim(actions),
                'rtgs': add_batch_dim(rtgs),
                'timesteps': add_batch_dim(timesteps),
                'moving_agent_mask': add_batch_dim(moving_agent_mask)
            })
            d['map'] = from_numpy({
                'road_points': add_batch_dim(road_points),
                'road_types': add_batch_dim(road_types)
            })
            d = MotionData(d)
            no_road_feats = False

        return d, no_road_feats


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    dset = RLWaymoDatasetCtRLSim(cfg, split_name='train')
    
    np.random.seed(10)
    random.seed(10)
    torch.manual_seed(10)
    
    dloader = DataLoader(dset, 
               batch_size=64, 
               shuffle=True, 
               num_workers=0,
               pin_memory=True,
               drop_last=True)

    i = 0
    for d in tqdm(dloader):
        i += 1
        break

if __name__ == '__main__':
    main()