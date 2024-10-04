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

class RLWaymoDataset(Dataset):
    # constants defining reward dimensions
    # 0: position target achieved (0 or 1)
    POS_TARGET_ACHIEVED_REW_IDX = 0
    # 1: heading target achieved (0 or 1)
    HEADING_TARGET_ACHIEVED_REW_IDX = 1
    # 2: speed target achieved (0 or 1)
    SPEED_TARGET_ACHIEVED_REW_IDX = 2
    # 3: position goal reward (shaped)
    POS_GOAL_SHAPED_REW_IDX = 3
    # 4: speed goal reward (shaped)
    SPEED_GOAL_SHAPED_REW_IDX = 4
    # 5: heading goal reward (shaped)
    HEADING_GOAL_SHAPED_REW_IDX = 5
    # 6: veh-veh collision reward (0 or 1)
    VEH_VEH_COLLISION_REW_IDX = 6
    # 7: veh-edge collision reward (0 or 1)
    VEH_EDGE_COLLISION_REW_IDX = 7
    
    def __init__(self, cfg, split_name='train', mode='train'):
        super(RLWaymoDataset, self).__init__()
        self.cfg = cfg
        self.cfg_dataset = cfg.dataset.waymo
        self.cfg_model = cfg.model
        self.data_root = self.cfg_dataset.dataset_path
        self.split_name = split_name 
        self.mode = mode
        
        # preprocess the nocturne waymo dataset for fast training
        if self.cfg_dataset.preprocess_real_data:
            self.files = glob.glob(os.path.join(self.data_root, f"{self.split_name}") + "/*.json")
        # preprocess the cat dataset for fast finetuning
        elif self.cfg_dataset.preprocess_simulated_data:
            self.files = glob.glob(self.cfg_dataset.simulated_dataset + "/*.json")
        # load the preprocessed data
        else:
            self.files = glob.glob(os.path.join(self.cfg_dataset.preprocess_dir, f"{self.split_name}") + "/*.pkl")
        
        self.files = sorted(self.files)
        self.dset_len = len(self.files)
        self.preprocess = self.cfg_dataset.preprocess
        
        if self.cfg_dataset.preprocess_simulated_data:
            self.preprocessed_dir = self.cfg_dataset.simulated_dataset_preprocessed_dir
        else:
            self.preprocessed_dir = os.path.join(self.cfg_dataset.preprocess_dir, f"{self.split_name}")

        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir, exist_ok=True)


    def get_roads(self, data):
        roads_data = data['roads']
        num_roads = len(roads_data)
        
        final_roads = []
        final_road_types = []
        road_edge_polylines = []
        for n in range(num_roads):
            curr_road_rawdat = roads_data[n]['geometry']
            if isinstance(curr_road_rawdat, dict):
                # for stop sign, repeat x/y coordinate along the point dimension
                final_roads.append(np.array((curr_road_rawdat['x'], curr_road_rawdat['y'], 1.0)).reshape(1, -1).repeat(self.cfg_dataset.max_num_road_pts_per_polyline, 0))
                final_road_types.append(get_road_type_onehot(roads_data[n]['type']))
            else:
                if roads_data[n]['type'] == 'road_edge':
                    polyline = []
                    for p in range(len(curr_road_rawdat)):
                        polyline.append(np.array((curr_road_rawdat[p]['x'], curr_road_rawdat[p]['y'])))
                    road_edge_polylines.append(np.array(polyline))
                
                # either we add points until we run out of points and append zeros
                # or we fill up with points until we reach max limit
                curr_road = []
                for p in range(len(curr_road_rawdat)):
                    curr_road.append(np.array((curr_road_rawdat[p]['x'], curr_road_rawdat[p]['y'], 1.0)))
                    if len(curr_road) == self.cfg_dataset.max_num_road_pts_per_polyline:
                        final_roads.append(np.array(curr_road))
                        curr_road = []
                        final_road_types.append(get_road_type_onehot(roads_data[n]['type']))
                if len(curr_road) < self.cfg_dataset.max_num_road_pts_per_polyline and len(curr_road) > 0:
                    tmp_curr_road = np.zeros((self.cfg_dataset.max_num_road_pts_per_polyline, 3))
                    tmp_curr_road[:len(curr_road)] = np.array(curr_road)
                    final_roads.append(tmp_curr_road)
                    final_road_types.append(get_road_type_onehot(roads_data[n]['type']))

        return np.array(final_roads), np.array(final_road_types), road_edge_polylines


    def extract_rawdata(self, agents_data):
        # Get indices of non-parked cars and cars that exist for the entire episode
        agent_data = []
        agent_types = []
        agent_actions = []
        agent_rewards = []
        agent_goals = []
        parked_agent_ids = []
        incomplete_ids = []
        last_exist_timesteps = []

        for n in range(len(agents_data)):
            ag_position = agents_data[n]['position']
            x_values = [entry['x'] for entry in ag_position]
            y_values = [entry['y'] for entry in ag_position]
            ag_position = np.column_stack((x_values, y_values))
            ag_heading = np.array(agents_data[n]['heading']).reshape((-1, 1))
            ag_velocity = agents_data[n]['velocity']
            x_values = [entry['x'] for entry in ag_velocity]
            y_values = [entry['y'] for entry in ag_velocity]
            ag_velocity = np.column_stack((x_values, y_values))
            # parked vehicle: average velocity < 0.05m/s
            if np.linalg.norm(ag_velocity, axis=-1).mean() < self.cfg_dataset.parked_car_velocity_threshold:
                parked_agent_ids.append(n)

            ag_existence = np.array(agents_data[n]['existence']).reshape((-1, 1))
            idx_of_first_disappearance = np.where(ag_existence == 0.0)[0]
            # once we find first missing step, all subsequent steps should be missing (as simulation is now undefined)
            if len(idx_of_first_disappearance) > 0:
                assert np.all(ag_existence[idx_of_first_disappearance[0]:] == 0.0)

            # only one timestep in ground-truth trajectory so no valid timesteps in offline RL dataset
            # since we need at least two timesteps in ground-truth trajectory to define a valid action with inverse Bicycle Model
            if len(idx_of_first_disappearance) > 0 and idx_of_first_disappearance[0] == 0:
                incomplete_ids.append(n)
                idx_of_last_existence = -1
            else:
                # for each agent, get the timestep of last existence
                idx_of_last_existence = np.where(ag_existence == 1.0)[0][-1]
                
            last_exist_timesteps.append(idx_of_last_existence)
            ag_actions = np.column_stack((agents_data[n]['acceleration'], agents_data[n]['steering']))

            ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
            ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
            agent_type = get_object_type_onehot(agents_data[n]['type'])
            # zero out reward for missing timesteps 
            rewards = np.array(agents_data[n]['reward']) * ag_existence 

            goal_position_x = agents_data[n]['goal_position']['x']
            goal_position_y = agents_data[n]['goal_position']['y']
            goal_heading = agents_data[n]['goal_heading']
            goal_speed = agents_data[n]['goal_speed']
            goal_velocity_x = goal_speed * np.cos(goal_heading)
            goal_velocity_y = goal_speed * np.sin(goal_heading)
            goal = np.array([goal_position_x, goal_position_y, goal_velocity_x, goal_velocity_y, goal_heading])
            goal = np.repeat(goal[None, :], len(ag_position), 0)

            ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
            agent_data.append(ag_state)
            agent_actions.append(ag_actions)
            agent_rewards.append(rewards)
            agent_types.append(agent_type)
            agent_goals.append(goal)
        
        # convert to numpy array
        agent_data = np.array(agent_data)
        agent_actions = np.array(agent_actions)
        agent_rewards = np.array(agent_rewards)
        agent_types = np.array(agent_types)
        agent_goals = np.array(agent_goals)
        parked_agent_ids = np.array(parked_agent_ids)
        incomplete_ids = np.array(incomplete_ids)
        last_exist_timesteps = np.array(last_exist_timesteps)
        
        return agent_data, agent_actions, agent_rewards, agent_types, agent_goals, parked_agent_ids, incomplete_ids, last_exist_timesteps


    def compute_dist_to_nearest_road_edge_rewards(self, ag_data, road_edge_polylines):
        # get all road edge polylines
        dist_to_road_edge_rewards = []
        for n in range(len(ag_data)):
            dist_to_road_edge = compute_distance_to_road_edge(ag_data[n, :, 0].reshape(1, -1),
                                                                ag_data[n, :, 1].reshape(1, -1), road_edge_polylines)
            dist_to_road_edge_rewards.append(-dist_to_road_edge / self.cfg_dataset.dist_to_road_edge_scaling_factor)
        
        dist_to_road_edge_rewards = np.array(dist_to_road_edge_rewards)
        
        return dist_to_road_edge_rewards 


    def compute_dist_to_nearest_vehicle_rewards(self, ag_data, normalize=True):
        num_timesteps = ag_data.shape[1]
        
        ag_positions = ag_data[:,:,:2]
        ag_existence = ag_data[:,:,-1]

        # set x/y position at each nonexisting timestep to np.inf
        mask = np.repeat(ag_existence[:,:,np.newaxis], repeats=2, axis=-1).astype(bool)
        ag_positions[~mask] = np.inf

        # data[:, np.newaxis] has shape (A, 1, 90, 2) and data[np.newaxis, :] has shape (1, A, 90, 2)
        # Subtracting these gives an array of shape (A, A, 90, 2) with pairwise differences
        diff = ag_positions[:, np.newaxis] - ag_positions[np.newaxis, :]
        squared_dist = np.sum(diff**2, axis=-1)

        # Replace zero distances (distance to self) with np.inf
        for i in range(num_timesteps):
            np.fill_diagonal(squared_dist[:,:,i], np.inf)

        # Find minimum distance for each agent at each timestep, shape (A, 90)
        dist_nearest_vehicle = np.sqrt(np.min(squared_dist, axis=1))
        # handles case when only one valid agent at specific timestep
        dist_nearest_vehicle[dist_nearest_vehicle == np.inf] = np.nan 
        
        # if dist > 15, give 15
        if normalize:
            dist_nearest_vehicle = np.clip(dist_nearest_vehicle * ag_existence, a_min=0.0, a_max=self.cfg_dataset.max_veh_veh_distance)
            # given that every reward is in [0, 15], we will normalize this to be between [0, 1] by simply dividing by 15.0
            dist_nearest_vehicle = dist_nearest_vehicle / self.cfg_dataset.max_veh_veh_distance
        else:
            dist_nearest_vehicle = dist_nearest_vehicle * ag_existence

        # set reward to 0 when undefined
        dist_nearest_vehicle = np.nan_to_num(dist_nearest_vehicle, nan=0.0)
        
        return dist_nearest_vehicle

    
    def compute_rewards(self, ag_data, ag_rewards, veh_edge_dist_rewards, veh_veh_dist_rewards):
        ag_existence = ag_data[:, :, -1:]

        processed_rewards = np.array(ag_rewards)
        if self.cfg_dataset.remove_shaped_goal:
            goal_pos_rewards = processed_rewards[:, :, self.POS_TARGET_ACHIEVED_REW_IDX] * self.cfg_dataset.pos_target_achieved_rew_multiplier
        else:
            goal_pos_rewards = processed_rewards[:, :, self.POS_TARGET_ACHIEVED_REW_IDX] * self.cfg_dataset.pos_target_achieved_rew_multiplier \
                 + (np.clip(processed_rewards[:, :, self.POS_GOAL_SHAPED_REW_IDX], a_min=self.cfg_dataset.pos_goal_shaped_min, a_max=self.cfg_dataset.pos_goal_shaped_max) - self.cfg_dataset.pos_goal_shaped_max) * (1 / self.cfg_dataset.pos_goal_shaped_max)
        goal_pos_rewards = goal_pos_rewards[:, :, np.newaxis] * ag_existence

        goal_heading_rewards = processed_rewards[:, :, self.HEADING_TARGET_ACHIEVED_REW_IDX] + processed_rewards[:, :, self.HEADING_GOAL_SHAPED_REW_IDX]
        goal_heading_rewards = goal_heading_rewards[:, :, np.newaxis] * ag_existence

        goal_velocity_rewards = processed_rewards[:, :, self.SPEED_TARGET_ACHIEVED_REW_IDX] + processed_rewards[:, :, self.SPEED_GOAL_SHAPED_REW_IDX]
        goal_velocity_rewards = goal_velocity_rewards[:, :, np.newaxis] * ag_existence
        
        if self.cfg_dataset.remove_shaped_veh_reward:
            veh_veh_collision_rewards = -1 * processed_rewards[:, :, self.VEH_VEH_COLLISION_REW_IDX] * self.cfg_dataset.veh_veh_collision_rew_multiplier
        else:
            veh_veh_collision_rewards = veh_veh_dist_rewards - \
                processed_rewards[:, :, self.VEH_VEH_COLLISION_REW_IDX] * self.cfg_dataset.veh_veh_collision_rew_multiplier

        veh_veh_collision_rewards = veh_veh_collision_rewards[:, :, np.newaxis] * ag_existence
        
        if self.cfg_dataset.remove_shaped_edge_reward:
            veh_edge_collision_rewards = -1 * processed_rewards[:, :, self.VEH_EDGE_COLLISION_REW_IDX] * self.cfg_dataset.veh_edge_collision_rew_multiplier
        else:
            veh_edge_collision_rewards = np.clip(np.abs(veh_edge_dist_rewards) * self.cfg_dataset.dist_to_road_edge_scaling_factor, a_min=0, a_max=5) / 5. - \
                processed_rewards[:, :, self.VEH_EDGE_COLLISION_REW_IDX] * self.cfg_dataset.veh_edge_collision_rew_multiplier

        veh_edge_collision_rewards = veh_edge_collision_rewards[:, :, np.newaxis] * ag_existence

        all_rewards = np.concatenate((goal_pos_rewards, goal_heading_rewards, goal_velocity_rewards,
                                    veh_veh_collision_rewards, veh_edge_collision_rewards), axis=-1)
        return all_rewards

    
    def select_relevant_agents(self, agent_states, agent_types, actions, rtgs, goals, origin_agent_idx, timestep, moving_agent_mask, relevant_agent_idxs=None):
        origin_states = agent_states[origin_agent_idx, timestep, :2].reshape(1, -1)
        dist_to_origin = np.linalg.norm(origin_states - agent_states[:, timestep, :2], axis=-1)
        valid_agents = np.where(dist_to_origin < self.cfg_dataset.agent_dist_threshold)[0]

        final_agent_states = np.zeros((self.cfg_dataset.max_num_agents, *agent_states[0].shape))
        final_agent_types = -np.ones((self.cfg_dataset.max_num_agents, *agent_types[0].shape))
        final_actions = np.zeros((self.cfg_dataset.max_num_agents, *actions[0].shape))
        final_rtgs = np.zeros((self.cfg_dataset.max_num_agents, *rtgs[0].shape))
        final_goals = np.zeros((self.cfg_dataset.max_num_agents, *goals[0].shape))
        final_moving_agent_mask = np.zeros(self.cfg_dataset.max_num_agents)

        if relevant_agent_idxs is None or len(relevant_agent_idxs) == 0:
            closest_ag_ids = np.argsort(dist_to_origin)[:self.cfg_dataset.max_num_agents]
            closest_ag_ids = np.intersect1d(closest_ag_ids, valid_agents)
            # shuffle ids so it is not ordered by distance
            if self.split_name == 'train':
                np.random.shuffle(closest_ag_ids)
        else:
            closest_ag_ids = np.array(relevant_agent_idxs).astype(int)
            closest_ag_ids = np.intersect1d(closest_ag_ids, valid_agents)
            
            if len(closest_ag_ids) < len(relevant_agent_idxs):
                out_of_range_vehicles = np.setdiff1d(relevant_agent_idxs, closest_ag_ids)
                relevant_agent_idxs = [idx for idx in relevant_agent_idxs if idx not in out_of_range_vehicles]
        
        final_agent_states[:len(closest_ag_ids)] = agent_states[closest_ag_ids]
        final_agent_types[:len(closest_ag_ids)] = agent_types[closest_ag_ids]
        final_actions[:len(closest_ag_ids)] = actions[closest_ag_ids]
        final_rtgs[:len(closest_ag_ids)] = rtgs[closest_ag_ids]
        final_goals[:len(closest_ag_ids)] = goals[closest_ag_ids]
        final_moving_agent_mask[:len(closest_ag_ids)] = moving_agent_mask[closest_ag_ids]

        # idx of origin agent in new state tensors
        new_origin_agent_idx = np.where(closest_ag_ids == origin_agent_idx)[0][0]
        if relevant_agent_idxs is None:
            return final_agent_states, final_agent_types, final_actions, final_rtgs, final_goals, final_moving_agent_mask, new_origin_agent_idx
        else:
            new_agent_idx_dict = {}
            for new_idx, old_idx in enumerate(closest_ag_ids):
                new_agent_idx_dict[old_idx] = new_idx
            return final_agent_states, final_agent_types, final_actions, final_rtgs, final_goals, final_moving_agent_mask, new_agent_idx_dict, relevant_agent_idxs


    def undiscretize_actions(self, actions):
        # Initialize the array for the continuous actions
        actions_shape = (actions.shape[0], actions.shape[1], 2)
        continuous_actions = np.zeros(actions_shape)
        
        # Separate the combined actions back into their discretized components
        continuous_actions[:, :, 0] = actions // self.cfg_dataset.steer_discretization  # Acceleration component
        continuous_actions[:, :, 1] = actions % self.cfg_dataset.steer_discretization   # Steering component
        
        # Reverse the discretization
        continuous_actions[:, :, 0] /= (self.cfg_dataset.accel_discretization - 1)
        continuous_actions[:, :, 1] /= (self.cfg_dataset.steer_discretization - 1)
        
        # Denormalize to get back the original continuous values
        continuous_actions[:, :, 0] = (continuous_actions[:, :, 0] * (self.cfg_dataset.max_accel - self.cfg_dataset.min_accel)) + self.cfg_dataset.min_accel
        continuous_actions[:, :, 1] = (continuous_actions[:, :, 1] * (self.cfg_dataset.max_steer - self.cfg_dataset.min_steer)) + self.cfg_dataset.min_steer
        
        return continuous_actions

    
    def get_tilt_logits(self, goal_tilt, veh_tilt, road_tilt):
        rtg_bin_values = np.zeros((self.cfg_dataset.rtg_discretization, 3))
        rtg_bin_values[:, 0] = goal_tilt * np.linspace(0, 1, self.cfg_dataset.rtg_discretization)
        rtg_bin_values[:, 1] = veh_tilt * np.linspace(0, 1, self.cfg_dataset.rtg_discretization)
        rtg_bin_values[:, 2] = road_tilt * np.linspace(0, 1, self.cfg_dataset.rtg_discretization)

        return rtg_bin_values


    def undiscretize_rtgs(self, rtgs):
        continuous_rtgs = np.zeros_like(rtgs).astype(float)
        continuous_rtgs[:, :, 0] = rtgs[:, :, 0] / (self.cfg_dataset.rtg_discretization - 1)
        continuous_rtgs[:, :, 1] = rtgs[:, :, 1] / (self.cfg_dataset.rtg_discretization - 1)

        continuous_rtgs[:, :, 0] = (continuous_rtgs[:, :, 0] * (self.cfg_dataset.max_rtg_pos - self.cfg_dataset.min_rtg_pos)) + self.cfg_dataset.min_rtg_pos
        continuous_rtgs[:, :, 1] = (continuous_rtgs[:, :, 1] * (self.cfg_dataset.max_rtg_veh - self.cfg_dataset.min_rtg_veh)) + self.cfg_dataset.min_rtg_veh
        continuous_rtgs[:, :, 2] = rtgs[:, :, 2] / (self.cfg_dataset.rtg_discretization - 1)
        continuous_rtgs[:, :, 2] = (continuous_rtgs[:, :, 2] * (self.cfg_dataset.max_rtg_road - self.cfg_dataset.min_rtg_road)) + self.cfg_dataset.min_rtg_road


        return continuous_rtgs

    
    def discretize_actions(self, actions):
        # normalize
        actions[:, :, 0] = ((np.clip(actions[:, :, 0], a_min=self.cfg_dataset.min_accel, a_max=self.cfg_dataset.max_accel) - self.cfg_dataset.min_accel)
                             / (self.cfg_dataset.max_accel - self.cfg_dataset.min_accel))
        actions[:, :, 1] = ((np.clip(actions[:, :, 1], a_min=self.cfg_dataset.min_steer, a_max=self.cfg_dataset.max_steer) - self.cfg_dataset.min_steer)
                             / (self.cfg_dataset.max_steer - self.cfg_dataset.min_steer))

        # discretize the actions
        actions[:, :, 0] = np.round(actions[:, :, 0] * (self.cfg_dataset.accel_discretization - 1))
        actions[:, :, 1] = np.round(actions[:, :, 1] * (self.cfg_dataset.steer_discretization - 1))

        # combine into a single categorical value
        combined_actions = actions[:, :, 0] * self.cfg_dataset.steer_discretization + actions[:, :, 1]

        return combined_actions

    
    def discretize_rtgs(self, rtgs):
        rtgs[:, :, 0] = np.round(rtgs[:, :, 0] * (self.cfg_dataset.rtg_discretization - 1))
        rtgs[:, :, 1] = np.round(rtgs[:, :, 1] * (self.cfg_dataset.rtg_discretization - 1))
        rtgs[:, :, 2] = np.round(rtgs[:, :, 2] * (self.cfg_dataset.rtg_discretization - 1))

        return rtgs

    
    def normalize_scene(self, agent_states, road_points, road_types, goals, origin_agent_idx):
        # normalize scene to ego vehicle (this includes agent states, goals, and roads)
        yaw = agent_states[origin_agent_idx, 0, 4]
        angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
        translation = agent_states[origin_agent_idx, 0, :2].copy()
        translation = translation[np.newaxis, np.newaxis, :]

        agent_states[:, :, :2] = apply_se2_transform(coordinates=agent_states[:, :, :2],
                                           translation=translation,
                                           yaw=angle_of_rotation)
        agent_states[:, :, 2:4] = apply_se2_transform(coordinates=agent_states[:, :, 2:4],
                                           translation=np.zeros_like(translation),
                                           yaw=angle_of_rotation)
        agent_states[:, :, 4] = angle_sub_tensor(agent_states[:, :, 4], -angle_of_rotation.reshape(1, 1))
        assert np.all(agent_states[:, :, 4] <= np.pi) and np.all(agent_states[:, :, 4] >= -np.pi)
        goals[:, :2] = apply_se2_transform(coordinates=goals[:, :2],
                                 translation=translation[:, 0],
                                 yaw=angle_of_rotation)
        if self.cfg_dataset.goal_dim == 5:
            goals[:, 2:4] = apply_se2_transform(coordinates=goals[:, 2:4],
                                    translation=np.zeros_like(translation[:, 0]),
                                    yaw=angle_of_rotation)
            goals[:, 4] = angle_sub_tensor(goals[:, 4], -angle_of_rotation.reshape(1))
        road_points[:, :, :2] = apply_se2_transform(coordinates=road_points[:, :, :2],
                                                    translation=translation,
                                                    yaw=angle_of_rotation)

        if len(road_points) > self.cfg_dataset.max_num_road_polylines:
            max_road_dist_to_orig = (np.linalg.norm(road_points[:, :, :2], axis=-1) * road_points[:, :, -1]).max(1)
            closest_roads_to_ego = np.argsort(max_road_dist_to_orig)[:self.cfg_dataset.max_num_road_polylines]
            final_road_points = road_points[closest_roads_to_ego]
            final_road_types = road_types[closest_roads_to_ego]
        else:
            final_road_points = np.zeros((self.cfg_dataset.max_num_road_polylines, *road_points.shape[1:]))
            final_road_points[:len(road_points)] = road_points
            final_road_types = -np.ones((self.cfg_dataset.max_num_road_polylines, road_types.shape[1]))
            final_road_types[:len(road_points)] = road_types
        
        return agent_states, final_road_points, final_road_types, goals


    def get_distance_to_road_edge(self, agent_states, road_feats):
        road_type = np.argmax(road_feats[:, 4:12], axis=1).astype(int)
        mask = road_type == 3
        road_feats = road_feats[mask]
        road_points = np.concatenate([road_feats[:, :2], road_feats[:, 2:4]], axis=0)
        agent_positions = agent_states[:, :, :2].reshape(-1, 2)
        # Compute the difference along each dimension [N, M, 2]
        diff = agent_positions[:, np.newaxis, :] - road_points[np.newaxis, :, :]
        # Compute the squared distances [N, M]
        squared_distances = np.sum(diff ** 2, axis=2)
        # Find the minimum squared distance for each point in array1 [N,]
        min_squared_distances = np.min(squared_distances, axis=1)
        # If you need the actual distances, take the square root
        min_distances = np.sqrt(min_squared_distances)

        min_distances = min_distances.reshape(24, 33)
        return min_distances

    
    def select_random_origin_agent(self, agent_states, moving_mask):
        pass

    
    def get_data(self, data, idx):
        pass
        

    def get(self, idx: int):
        # search for file with at least 2 agents
        if not self.cfg_dataset.preprocess:
            proceed = False 
            while not proceed:
                with open(self.files[idx], 'r') as file:
                    data = json.load(file)
                    # search for scenes with at least 2 agents
                    if len(data['objects']) == 1:
                        idx += 1
                    else:
                        proceed = True 
            
            d, no_road_feats = self.get_data(data, idx)

        else:
            proceed = False 
            while not proceed:
                raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
                raw_path = os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl')
                if os.path.exists(raw_path):
                    with open(raw_path, 'rb') as f:
                        data = pickle.load(f)
                    proceed = True
                else:
                    idx += 1

                if proceed:
                    if self.mode =='train':
                        d, no_road_feats = self.get_data(data, idx)
                        # only load sample if it has a map
                        if no_road_feats:
                            proceed = False 
                            idx += 1
                    else:
                        rtgs, road_points, road_types = self.get_data(data, idx)
                        d = {
                            'rtgs': rtgs,
                            'road_points': road_points,
                            'road_types': road_types
                        }
        
        return d

    
    def len(self):
        return self.dset_len