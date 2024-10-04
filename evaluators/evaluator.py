import json
import os
import sys
import time
import argparse
import pickle
import random

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import nocturne
import pdb
from tqdm import tqdm
from scipy.spatial import distance
from cfgs.config import set_display_window
from utils.sim import *
from utils.data import get_object_type_str, compute_distance_to_road_edge
from datasets.rl_waymo import RLWaymoDatasetCtRLSim, RLWaymoDatasetCTGPlusPlus
from nocturne.bicycle_model import BicycleModel

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_rl_waymo = self.cfg.dataset.waymo
        self.steps = self.cfg.nocturne.steps
        self.dt = self.cfg.nocturne.dt
        self.history_steps = self.cfg.nocturne.history_steps


    def load_scenario(self, file_path, file):
        sim = get_sim(self.cfg, file_path, self.test_filenames, file)
        scenario = sim.getScenario()
        vehicles = scenario.vehicles()
        for veh in vehicles:
            veh.expert_control = False 
            veh.physics_simulated = True
        
        return sim, scenario, vehicles


    def load_preprocessed_data(self, file_path, file):
        file_exists = False 
        filename = os.path.join(file_path, f'{self.test_filenames[file][:-5]}_physics.pkl')
        
        if filename in self.preprocessed_dset.files:
            file_exists = True

        if file_exists:
            idx = self.preprocessed_dset.files.index(filename)
            preproc_data = self.preprocessed_dset[idx]
        else:
            preproc_data = None
        
        return preproc_data, file_exists


    def initialize_goal_dict(self, veh, gt_traj_data):
        goal_pos = np.array([veh.target_position.x, veh.target_position.y])
        goal_heading = veh.target_heading
        goal_speed = veh.target_speed
        idx_disappear = np.where(gt_traj_data[:, 4] == 0)[0]
        if len(idx_disappear) > 0:
            idx_goal = idx_disappear[0] - 1
            if np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj_data[idx_goal, :2]
                goal_heading = gt_traj_data[idx_goal, 2]
                goal_speed = gt_traj_data[idx_goal, 3]
        
        return {
            'pos': goal_pos,
            'heading': goal_heading,
            'speed': goal_speed
        }


    def compute_goal_dist_normalizer(self, veh, goal_pos):
        # Precompute goal-dist normalizer (used for reward computation)
        obj_pos = veh.getPosition()
        obj_pos = np.array([obj_pos.x, obj_pos.y])
        dist = np.linalg.norm(obj_pos - goal_pos)
        return dist

    
    def compute_nearest_dist_all(self, t, vehicle_data_dict):
        all_x = np.array([vehicle_data_dict[v]["position"][t]['x'] for v in vehicle_data_dict.keys()])
        all_y = np.array([vehicle_data_dict[v]["position"][t]['y'] for v in vehicle_data_dict.keys()])
        all_existence = np.array([vehicle_data_dict[v]["existence"][t] for v in vehicle_data_dict.keys()])
        ag_data = np.concatenate([all_x[:, np.newaxis], all_y[:, np.newaxis], all_existence[:, np.newaxis]], axis=1)[:, np.newaxis, :]
        veh_veh_dist_rewards = self.preprocessed_dset.compute_dist_to_nearest_vehicle_rewards(ag_data, normalize=False) * all_existence[:, np.newaxis].astype(float)

        all_gt_x = np.array([vehicle_data_dict[v]["gt_position"][t]['x'] for v in vehicle_data_dict.keys()])
        all_gt_y = np.array([vehicle_data_dict[v]["gt_position"][t]['y'] for v in vehicle_data_dict.keys()])
        gt_ag_data = np.concatenate([all_gt_x[:, np.newaxis], all_gt_y[:, np.newaxis], all_existence[:, np.newaxis]], axis=1)[:, np.newaxis, :]
        veh_veh_dist_rewards_gt = self.preprocessed_dset.compute_dist_to_nearest_vehicle_rewards(gt_ag_data, normalize=False) * all_existence[:, np.newaxis].astype(float)

        for i, veh_id in enumerate(vehicle_data_dict.keys()):
            vehicle_data_dict[veh_id]["nearest_dist"].append(veh_veh_dist_rewards[i, 0])
            vehicle_data_dict[veh_id]["gt_nearest_dist"].append(veh_veh_dist_rewards_gt[i, 0])

        return vehicle_data_dict

    
    def compute_dense_reward(self, t, vehicle_data_dict, road_edge_polylines):
        # get into right format to call dist to road edge function
        all_x = np.array([vehicle_data_dict[v]["position"][t]['x'] for v in vehicle_data_dict.keys()])
        all_y = np.array([vehicle_data_dict[v]["position"][t]['y'] for v in vehicle_data_dict.keys()])
        all_existence = np.array([vehicle_data_dict[v]["existence"][t] for v in vehicle_data_dict.keys()])
        processed_rewards = np.array([vehicle_data_dict[v]['reward'] for v in vehicle_data_dict.keys()])
        processed_rewards = processed_rewards * all_existence[:, np.newaxis, np.newaxis].astype(float)
        
        ag_data = ag_data = np.concatenate([all_x[:, np.newaxis], all_y[:, np.newaxis]], axis=1)[:, np.newaxis, :]
        veh_edge_dist_rewards = self.preprocessed_dset.compute_dist_to_nearest_road_edge_rewards(ag_data, road_edge_polylines)
        veh_edge_dist_rewards = veh_edge_dist_rewards * all_existence[:, np.newaxis].astype(float)
        
        # get into right format to call dist to nearest vehicle function
        ag_data = np.concatenate([all_x[:, np.newaxis], all_y[:, np.newaxis], all_existence[:, np.newaxis]], axis=1)[:, np.newaxis, :]
        veh_veh_dist_rewards = self.preprocessed_dset.compute_dist_to_nearest_vehicle_rewards(ag_data, normalize=False) * all_existence[:, np.newaxis].astype(float)
        
        all_gt_x = np.array([vehicle_data_dict[v]["gt_position"][t]['x'] for v in vehicle_data_dict.keys()])
        all_gt_y = np.array([vehicle_data_dict[v]["gt_position"][t]['y'] for v in vehicle_data_dict.keys()])
        gt_ag_data = np.concatenate([all_gt_x[:, np.newaxis], all_gt_y[:, np.newaxis], all_existence[:, np.newaxis]], axis=1)[:, np.newaxis, :]
        veh_veh_dist_rewards_gt = self.preprocessed_dset.compute_dist_to_nearest_vehicle_rewards(gt_ag_data, normalize=False) * all_existence[:, np.newaxis].astype(float)
            
        for i, veh_id in enumerate(vehicle_data_dict.keys()):
            vehicle_data_dict[veh_id]["nearest_dist"].append(veh_veh_dist_rewards[i, 0] * self.cfg_rl_waymo.max_veh_veh_distance)
            vehicle_data_dict[veh_id]["gt_nearest_dist"].append(veh_veh_dist_rewards_gt[i, 0] * self.cfg_rl_waymo.max_veh_veh_distance)
        
        veh_veh_dist_rewards = np.clip(veh_veh_dist_rewards, a_min=0.0, a_max=self.cfg_rl_waymo.max_veh_veh_distance)
        veh_veh_dist_rewards = veh_veh_dist_rewards / self.cfg_rl_waymo.max_veh_veh_distance
        
        all_rewards = self.preprocessed_dset.compute_rewards(ag_data, processed_rewards, veh_edge_dist_rewards, veh_veh_dist_rewards)
        all_rewards = np.concatenate([all_rewards[:, :, :1], all_rewards[:, :, 3:]], axis=-1)
        
        for i, veh_id in enumerate(vehicle_data_dict.keys()):
            vehicle_data_dict[veh_id]["dense_reward"].append(all_rewards[i, 0])
        
        return vehicle_data_dict


    def extract_road_edge_polylines(self, roads_data):
        num_roads = len(roads_data)
        road_edge_polylines = []
        for n in range(num_roads):
            curr_road_rawdat = roads_data[n]['geometry']
            if isinstance(curr_road_rawdat, dict):
                continue 
            
            if roads_data[n]['type'] == 'road_edge':
                polyline = []
                for p in range(len(curr_road_rawdat)):
                    polyline.append(np.array((curr_road_rawdat[p]['x'], curr_road_rawdat[p]['y'])))
                road_edge_polylines.append(np.array(polyline))
        
        return road_edge_polylines

    
    def apply_gt_action(self, veh, t, gt_data_dict, vehicle_data_dict):
        veh_id = veh.getID()
        # action is only defined if state at next timestep is defined
        veh_exists = gt_data_dict[veh_id]['traj'][t][4] and gt_data_dict[veh_id]['traj'][t+1][4]
        # once we encounter the first missing timestep, all future timesteps are also missing
        # this is because we need contiguous sequence to push through nocturne simulator
        if t > 0 and vehicle_data_dict[veh_id]["existence"][-1] == 0:
            veh_exists = 0
        
        if not veh_exists:
            acceleration = 0.0
            steering = 0.0
            veh.setPosition(-1000000, -1000000)  # make cars disappear if they are out of actions
        else:
            bike_model = BicycleModel(x=gt_data_dict[veh_id]['traj'][t+1][0],
                                      y=gt_data_dict[veh_id]['traj'][t+1][1],
                                      theta=gt_data_dict[veh_id]['traj'][t+1][2],
                                      vel=gt_data_dict[veh_id]['traj'][t+1][3],
                                      L=gt_data_dict[veh_id]['traj'][t+1][-1],
                                      dt=self.dt)
            
            acceleration, steering, _, _ = bike_model.backward(prev_pos=np.array([veh.getPosition().x,veh.getPosition().y]), 
                                                     prev_theta=veh.getHeading(),
                                                     prev_vel=veh.getSpeed())
        
        veh_action = [acceleration, steering]
        
        if acceleration > 0.0:
            veh.acceleration = acceleration
        else:
            veh.brake(np.abs(acceleration))
        veh.steering = steering

        return veh, veh_action

    
    def reset(self):
        pass

    
    def initialize_vehicle_data_dict(self, veh, goal_dict):
        pass
    
    
    def update_vehicle_data_dict(self, t, vehicles, vehicle_data_dict, goal_dict, goal_dist_normalizer, gt_data_dict, preproc_data, road_edge_polylines):
        pass

    
    def update_running_statistics(self, data_dict):
        pass

    
    def compute_metrics(self):
        pass

    

    
    


