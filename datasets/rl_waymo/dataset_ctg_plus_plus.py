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

class RLWaymoDatasetCTGPlusPlus(RLWaymoDataset):
    def __init__(self, cfg, split_name='train', mode='train'):
        super(RLWaymoDatasetCTGPlusPlus, self).__init__(cfg, split_name, mode)


    def select_random_origin_agent(self, agent_states, moving_mask):
        # search for moving agent that exists at future timestep
        valid_idxs = np.where((agent_states[:, self.cfg_dataset.input_horizon, -1] == 1) * moving_mask)[0]
        if len(valid_idxs) == 0:
            return 0, False
        rand_idx = np.random.choice(len(valid_idxs))
        return valid_idxs[rand_idx], True

    
    def _get_constant_velocity_futures(self, present_states):
        # Shape [A, 13]
        # [0: local_pos_x, 
        #  1: local_pos_y,
        #  2: local_vel_x,
        #  3: local_vel_y,
        #  4: local_yaw,
        #  5: global_pos_x,
        #  6: global_pos_y,
        #  7: global_vel_x,
        #  8: global_vel_y,
        #  9: global_yaw,
        #  10: length,
        #  11: width,
        #  12: existence]
        present_states = present_states.copy()
        
        T = self.cfg_dataset.train_context_length - 10
        cv_future_states = np.zeros((present_states.shape[0], T, present_states.shape[1]))
        cv_future_states[:, :, 2:5] = np.expand_dims(present_states[:, 2:5], axis=1).repeat(T, axis=1)
        cv_future_states[:, :, 7:] = np.expand_dims(present_states[:, 7:], axis=1).repeat(T, axis=1)

        dt = 0.1
        offset = np.expand_dims(np.arange(T), axis=0) * dt + dt
        offset_local_x = offset * present_states[:, 2:3].repeat(T, axis=1)
        offset_local_y = offset * present_states[:, 3:4].repeat(T, axis=1)
        offset_global_x = offset * present_states[:, 7:8].repeat(T, axis=1)
        offset_global_y = offset * present_states[:, 8:9].repeat(T, axis=1)

        cv_future_states[:, :, 0] = present_states[:, :1].repeat(T, axis=1) + offset_local_x
        cv_future_states[:, :, 1] = present_states[:, 1:2].repeat(T, axis=1) + offset_local_y
        cv_future_states[:, :, 5] = present_states[:, 5:6].repeat(T, axis=1) + offset_global_x
        cv_future_states[:, :, 6] = present_states[:, 6:7].repeat(T, axis=1) + offset_global_y
        
        return cv_future_states



    def _prepare_relative_encodings(self, in_agents, present_states):
        relative_encoding_dimension = 7
        relative_encodings = np.zeros((in_agents.shape[0], in_agents.shape[0], in_agents.shape[1], relative_encoding_dimension))

        global_yaw_dimension = 9
        present_headings = present_states[:, 0, global_yaw_dimension].copy()

        # Using broadcasting for rotation_matrices
        cosines = np.cos(-present_headings + np.pi/2)
        sines = np.sin(-present_headings + np.pi/2)

        # [N_agents, 2, 2]
        rotation_matrices = np.array([
            [cosines, -sines],
            [sines, cosines]
        ]).transpose(2, 0, 1)

        global_positions_all = in_agents[:, :, 5:7].copy()
        present_positions_all = present_states[:, 0, 5:7].copy()
        global_yaws_all = in_agents[:, :, 9].copy()
        present_yaws_all = present_states[:, 0, 9].copy()
        global_speeds_all = np.linalg.norm(in_agents[:, :, 2:4], ord=2, axis=-1)
        present_speeds_all = np.linalg.norm(present_states[:, :1, 2:4], ord=2, axis=-1)

        for i in range(rotation_matrices.shape[0]):
            rotation_matrix = rotation_matrices[i]
            offsets = global_positions_all - present_positions_all[i]

            # Shape (n_agents, T, 2): This contains the offsets (x^j_t - x^i_0 , y^j_t - y^i_0) R^i_0.T 
            # so that the offsets are in i's local frame.
            rotated_offsets = np.matmul(offsets, rotation_matrix.T)
            relative_encodings[i, :, :, :2] = rotated_offsets

            yaw_offsets = global_yaws_all - present_yaws_all[i]
            relative_encodings[i, :, :, 2] = np.cos(yaw_offsets)
            relative_encodings[i, :, :, 3] = np.sin(yaw_offsets)

            relative_encodings[i, :, :, 4] = global_speeds_all * relative_encodings[i, :, :, 3] - present_speeds_all[i, 0]
            relative_encodings[i, :, :, 5] = global_speeds_all * relative_encodings[i, :, :, 4]

        relative_encodings[:, :, :, 6] = np.linalg.norm(np.expand_dims(global_positions_all, 0) - np.expand_dims(global_positions_all, 1), ord=2, axis=-1)

        return relative_encodings
    
    def select_indiv_agent_roads(self, agent_states, road_points, road_types):
        num_agents = agent_states.shape[0]
        if len(road_points) > self.cfg_dataset.max_num_road_polylines:   
            road_existence = road_points[None, :, :, -1].copy()
            road_existence[np.where(road_existence == 0.)] = np.nan
            max_dist_to_road = np.nanmax(np.linalg.norm(road_points[None, :, :, :2] - 
                                            agent_states[:, -1:, None, :2], axis=-1) * road_existence, axis=2)
            closest_roads_to_agent = np.argsort(max_dist_to_road, axis=-1)[:, :self.cfg_dataset.max_num_road_polylines]
            repeated_road_points = np.repeat(road_points[None], num_agents, axis=0)
            final_road_points = np.take_along_axis(repeated_road_points, closest_roads_to_agent[:, :, None, None], axis=1)
            repeated_road_types = np.repeat(road_types[None], num_agents, axis=0)
            final_road_types = np.take_along_axis(repeated_road_types, closest_roads_to_agent[:, : , None], axis=1)
        else:
            final_road_points = np.zeros((num_agents, self.cfg_dataset.max_num_road_polylines, *road_points.shape[1:]))
            final_road_points[:, :len(road_points)] = road_points[None]
            final_road_types = -np.ones((num_agents, self.cfg_dataset.max_num_road_polylines, road_types.shape[1]))
            final_road_types[:, :len(road_points)] = road_types[None]

        final_road_points[:, :, :, -1] = agent_states[:, -1:, -1:] * final_road_points[:, :, :, -1]
        final_road_types = final_road_types * agent_states[:, -1:, -1:]

        return final_road_points, final_road_types

    
    def _get_agents_local_frame_eval(self, agent_pasts, road_points, goals):
        num_agents = agent_pasts.shape[0]
        new_agents_pasts = np.zeros((agent_pasts.shape[0], agent_pasts.shape[1], agent_pasts.shape[2] + 5))
        new_agents_pasts[:, :, 5:] = agent_pasts

        new_roads = road_points.copy()
        # other agents
        for n in range(num_agents):
            if not agent_pasts[n, -1, -1]:
                continue

            yaw = agent_pasts[n, -1, 4]
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            translation = agent_pasts[n, -1, :2]

            new_agents_pasts[n, :, :2] = apply_se2_transform(coordinates=agent_pasts[n, :, :2], 
                                                             translation=translation.reshape(1, -1),
                                                             yaw=angle_of_rotation)
            new_agents_pasts[n, :, 2:4] = apply_se2_transform(coordinates=agent_pasts[n, :, 2:4], 
                                                              translation=np.zeros_like(translation).reshape(1, -1),
                                                              yaw=angle_of_rotation)
            new_agents_pasts[n, :, 4] = angle_sub_tensor(agent_pasts[n, :, 4], -angle_of_rotation)

            new_roads[n, :, :, :2] = apply_se2_transform(coordinates=road_points[n, :, :, :2],
                                                         translation=translation.reshape(1, 1, -1),
                                                         yaw=angle_of_rotation)
            new_roads[n][np.where(new_roads[n, :, :, -1] == 0)] = 0.0

            goals[n:n+1, :2] = apply_se2_transform(coordinates=goals[n:n+1, :2],
                                               translation=translation.reshape(1, -1),
                                               yaw=angle_of_rotation)
            if self.cfg_dataset.goal_dim == 5:
                goals[n:n+1, 2:4] = apply_se2_transform(coordinates=goals[n:n+1, 2:4],
                                        translation=np.zeros_like(translation).reshape(1, -1),
                                        yaw=angle_of_rotation)
                goals[n:n+1, 4] = angle_sub_tensor(goals[n:n+1, 4], -angle_of_rotation)
        
        return new_agents_pasts, new_roads, goals

    
    def _get_agents_local_frame(self, agent_pasts, agent_futures, road_points, goals):
        num_agents = agent_pasts.shape[0]
        new_agents_pasts = np.zeros((agent_pasts.shape[0], agent_pasts.shape[1], agent_pasts.shape[2] + 5))
        new_agents_pasts[:, :, 5:] = agent_pasts

        new_agents_futures = np.zeros((agent_futures.shape[0], agent_futures.shape[1], agent_futures.shape[2] + 5))
        new_agents_futures[:, :, 5:] = agent_futures

        new_roads = road_points.copy()
        # other agents
        for n in range(num_agents):
            if not agent_pasts[n, -1, -1]:
                continue

            yaw = agent_pasts[n, -1, 4]
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            translation = agent_pasts[n, -1, :2]

            new_agents_pasts[n, :, :2] = apply_se2_transform(coordinates=agent_pasts[n, :, :2], 
                                                             translation=translation.reshape(1, -1),
                                                             yaw=angle_of_rotation)
            new_agents_pasts[n, :, 2:4] = apply_se2_transform(coordinates=agent_pasts[n, :, 2:4], 
                                                              translation=np.zeros_like(translation).reshape(1, -1),
                                                              yaw=angle_of_rotation)
            new_agents_pasts[n, :, 4] = angle_sub_tensor(agent_pasts[n, :, 4], -angle_of_rotation)

            new_agents_futures[n, :, :2] = apply_se2_transform(coordinates=agent_futures[n, :, :2], 
                                                               translation=translation.reshape(1, -1),
                                                               yaw=angle_of_rotation)
            new_agents_futures[n, :, 2:4] = apply_se2_transform(coordinates=agent_futures[n, :, 2:4], 
                                                                translation=np.zeros_like(translation).reshape(1, -1),
                                                                yaw=angle_of_rotation)
            new_agents_futures[n, :, 4] = angle_sub_tensor(agent_futures[n, :, 4], -angle_of_rotation)

            new_roads[n, :, :, :2] = apply_se2_transform(coordinates=road_points[n, :, :, :2],
                                                         translation=translation.reshape(1, 1, -1),
                                                         yaw=angle_of_rotation)
            new_roads[n][np.where(new_roads[n, :, :, -1] == 0)] = 0.0

            goals[n:n+1, :2] = apply_se2_transform(coordinates=goals[n:n+1, :2],
                                               translation=translation.reshape(1, -1),
                                               yaw=angle_of_rotation)
            if self.cfg_dataset.goal_dim == 5:
                goals[n:n+1, 2:4] = apply_se2_transform(coordinates=goals[n:n+1, 2:4],
                                        translation=np.zeros_like(translation).reshape(1, -1),
                                        yaw=angle_of_rotation)
                goals[n:n+1, 4] = angle_sub_tensor(goals[n:n+1, 4], -angle_of_rotation)
        
        return new_agents_pasts, new_agents_futures, new_roads, goals

    def _normalize_actions(self, actions):
        actions[:, :, 0] = ((np.clip(actions[:, :, 0], a_min=self.cfg_dataset.min_accel, a_max=self.cfg_dataset.max_accel) - self.cfg_dataset.min_accel)
                             / (self.cfg_dataset.max_accel - self.cfg_dataset.min_accel))
        actions[:, :, 1] = ((np.clip(actions[:, :, 1], a_min=self.cfg_dataset.min_steer, a_max=self.cfg_dataset.max_steer) - self.cfg_dataset.min_steer)
                             / (self.cfg_dataset.max_steer - self.cfg_dataset.min_steer))
        
        return (2.0 * actions) - 1.0

    def _unnormalize_actions(self, actions):
        actions = (actions + 1.0) / 2.0
        actions[:, :, 0] = actions[:, :, 0] * (self.cfg_dataset.max_accel - self.cfg_dataset.min_accel) + self.cfg_dataset.min_accel
        actions[:, :, 1] = actions[:, :, 1] * (self.cfg_dataset.max_steer - self.cfg_dataset.min_steer) + self.cfg_dataset.min_steer
        
        return actions

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
        # max_timestep = np.max(last_exist_timesteps[moving_ids]) - (self.cfg_dataset.train_context_length - 1)
        max_timestep = np.max(last_exist_timesteps[moving_ids]) - (self.cfg_dataset.input_horizon + 1)
        # In this case, there will be some future timesteps such that all agents do not exist
        if max_timestep < 0:
            max_timestep = 0
        origin_t = np.random.randint(0, max_timestep+1)

        timesteps = np.arange(self.cfg_dataset.train_context_length) + origin_t
        timesteps = np.repeat(timesteps[np.newaxis, :, np.newaxis], self.cfg_dataset.max_num_agents, 0)
        timesteps = np.ones_like(timesteps) * (origin_t + self.cfg_dataset.input_horizon - 1)  # ultimate laziness...
        agent_states = ag_data[filtered_ag_ids, origin_t:origin_t+self.cfg_dataset.train_context_length]
        agent_types = ag_types[filtered_ag_ids]
        if origin_t == 0:
            _actions = ag_actions[filtered_ag_ids, origin_t:origin_t+self.cfg_dataset.train_context_length-1]
            actions = np.concatenate((np.zeros((len(filtered_ag_ids), 1, _actions.shape[-1])), _actions), axis=1)
        else:
            actions = ag_actions[filtered_ag_ids, origin_t-1:origin_t+self.cfg_dataset.train_context_length-1]
        rtgs = rtgs[filtered_ag_ids, origin_t:origin_t+self.cfg_dataset.train_context_length]

        # padding incomplete scenes
        current_num_timesteps = agent_states.shape[1]
        if current_num_timesteps < self.cfg_dataset.train_context_length:
            # states
            padded_agent_states = np.zeros((len(filtered_ag_ids), self.cfg_dataset.train_context_length, agent_states.shape[-1]))
            padded_agent_states[:, :current_num_timesteps] = agent_states
            agent_states = padded_agent_states.copy()
            # actions
            padded_agent_actions = np.zeros((len(filtered_ag_ids), self.cfg_dataset.train_context_length, actions.shape[-1]))
            padded_agent_actions[:, :current_num_timesteps] = actions[:, :-1]  # this -1 is to account for timestep offset.
            actions = padded_agent_actions.copy()
            # rtgs
            padded_agent_rtgs = np.zeros((len(filtered_ag_ids), self.cfg_dataset.train_context_length, rtgs.shape[-1]))
            padded_agent_rtgs[:, :current_num_timesteps] = rtgs
            rtgs = padded_agent_rtgs.copy()

        # filter for agents that move at least 0.05 metres
        moving_agent_mask = np.isin(filtered_ag_ids, moving_ids)
        # randomly choose moving agent to be at origin
        # if the scene does not contain any agent with >10 timesteps of existence, scene is invalid
        origin_agent_idx, valid_scene = self.select_random_origin_agent(agent_states, moving_agent_mask)
        
        if valid_scene:
            agent_states, agent_types, actions, rtgs, goals, moving_agent_mask, new_origin_agent_idx = self.select_relevant_agents(agent_states, agent_types, actions, rtgs, goals, origin_agent_idx, 0, moving_agent_mask)
        
        num_polylines = len(road_points)
        if num_polylines == 0 or not valid_scene:
            d = MotionData({})
            no_road_feats = True 
        else:
            # agent_states, road_points, road_types, goals = self.normalize_scene(agent_states, road_points, road_types, goals, new_origin_agent_idx)
            agent_states_past = agent_states[:, :self.cfg_dataset.input_horizon]
            agent_states_future = agent_states[:, self.cfg_dataset.input_horizon:]
            road_points, road_types = self.select_indiv_agent_roads(agent_states_past, road_points, road_types)

            yaw = agent_states_past[:, -1, 4].copy()
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            translation = agent_states_past[:, -1, :2].copy()
            translation_yaws = np.concatenate((translation, angle_of_rotation[:, None]), axis=-1)
            
            agent_states_past, agent_states_future, road_points, goals =\
                self._get_agents_local_frame(agent_states_past, agent_states_future, road_points, goals)
            
            past_relative_encoding = self._prepare_relative_encodings(agent_states_past, agent_states_past[:, -1:, :])
            if self.cfg_dataset.future_relative_encoding:
                if self.mode == 'train':
                    future_relative_encoding = self._prepare_relative_encodings(agent_states_future, agent_states_past[:, -1:, :])
                else:
                    # use rel_ag_states_future here to compare
                    cv_future_states = self._get_constant_velocity_futures(agent_states_past[:, -1])
                    future_relative_encoding = self._prepare_relative_encodings(cv_future_states, agent_states_past[:, -1:, :])
            else:
                future_relative_encoding = past_relative_encoding[:, :, -1:].repeat(agent_states_future.shape[1], axis=2)

            # remove global coodinates
            agent_states_past = np.concatenate((agent_states_past[:, :, 0:5], agent_states_past[:, :, 10:]), axis=-1)
            agent_states_future = np.concatenate((agent_states_future[:, :, 0:5], agent_states_future[:, :, -1:]), axis=-1)

            # apply normalization for diffusion, approx around 0.0
            agent_states_past[:, :, :2] /= self.cfg_dataset.state_normalizer.pos_div
            agent_states_past[:, :, 2:4] /= self.cfg_dataset.state_normalizer.vel_div
            agent_states_future[:, :, :2] /= self.cfg_dataset.state_normalizer.pos_div
            agent_states_future[:, :, 2:4] /= self.cfg_dataset.state_normalizer.vel_div
            goals[:, :2] /= self.cfg_dataset.state_normalizer.pos_div
            goals[:, 2:4] /= self.cfg_dataset.state_normalizer.vel_div
            road_points[:, :, :, :2] /= self.cfg_dataset.state_normalizer.pos_div

            actions = self._normalize_actions(actions)
            agent_actions_past = actions[:, :self.cfg_dataset.input_horizon]
            agent_actions_future = actions[:, self.cfg_dataset.input_horizon:]

            rtgs = self.discretize_rtgs(rtgs)
            rtgs = rtgs[:, :self.cfg_dataset.input_horizon]

            d = dict()
            d['idx'] = idx
            # need to add batch dim as pytorch_geometric batches along first dimension of torch Tensors
            d['agent'] = from_numpy({
                'agent_past_states': add_batch_dim(agent_states_past),
                'agent_past_actions': add_batch_dim(agent_actions_past),
                'agent_future_states': add_batch_dim(agent_states_future),
                'agent_future_actions': add_batch_dim(agent_actions_future),
                'past_relative_encodings': add_batch_dim(past_relative_encoding),
                'future_relative_encodings': add_batch_dim(future_relative_encoding),
                'agent_types': add_batch_dim(agent_types), 
                'goals': add_batch_dim(goals),
                'rtgs': add_batch_dim(rtgs),
                'timesteps': add_batch_dim(timesteps),
                'moving_agent_mask': add_batch_dim(moving_agent_mask),
                'agent_translation_yaws': add_batch_dim(translation_yaws)
            })
            d['map'] = from_numpy({
                'road_points': add_batch_dim(road_points),
                'road_types': add_batch_dim(road_types)
            })
            d = MotionData(d)
            no_road_feats = False

        return d, no_road_feats