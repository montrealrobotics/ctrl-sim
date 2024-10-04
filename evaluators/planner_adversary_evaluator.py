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
from evaluators.evaluator import Evaluator

class PlannerAdversaryEvaluator(Evaluator):
    def __init__(self, cfg, planner, adversary):
        super(PlannerAdversaryEvaluator, self).__init__(cfg)

        self.planner = planner 
        self.adversary = adversary

        with open(cfg.cat.dict_path, 'rb') as f:
            eval_planner_dict = pickle.load(f)
        self.test_filenames = []
        self.file_to_id = {}
        for k in eval_planner_dict:
            filename = eval_planner_dict[k]['nocturne_path'][66:]
            self.test_filenames.append(filename)
            self.file_to_id[filename] = k
        self.eval_planner_dict = eval_planner_dict
        self.ego_vehicle = None
        self.adversary_vehicle = None
        self.preprocessed_dset = RLWaymoDatasetCtRLSim(self.cfg, split_name='val_interactive', mode='eval')
        self.reset()


    def reset(self):
        seed = self.cfg.eval_planner_adversary.seed
        random.seed(seed)  # Python random module.
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  # PyTorch.
        torch.cuda.manual_seed(seed)  # PyTorch, for CUDA.
        
        # ego planner metrics
        self.ades_all = []
        self.fdes_all = []
        self.goal_achieved_all = []
        self.progress_all = []
        self.collision_rate_scenario = []
        self.collision_rate_w_adv_scenario = []
        self.offroad_rate_scenario = []
        self.ego_jerk_all = []
        self.ego_steering_rate_all = []
        self.ego_accel_all = []
        
        # adversary metrics
        self.lin_speed_sim_all = []
        self.lin_speed_gt_all = []
        self.ang_speed_sim_all = []
        self.ang_speed_gt_all = []
        self.accel_sim_all = []
        self.accel_gt_all = []
        self.nearest_dist_sim_all = []
        self.nearest_dist_gt_all = []
        self.collision_speed_with_ego = []


    def initialize_vehicle_data_dict(self, veh, goal_dict):
        return {
            "gt_position": [], # [{'x': float, 'y': float}, ...]
            "gt_speed": [], # [float, ...]
            "gt_heading": [], # [float, ...]
            "gt_acceleration": [], # [float, ...]
            "gt_nearest_dist": [], # [float, ...]
            "position": [], # [{'x': float, 'y': float}, ...]
            "velocity": [],  # [{'x': float, 'y': float}, ...]
            "heading": [], # [float, ...]
            "nearest_dist": [], # [float, ...]
            "existence": [],
            "acceleration": [], # [float, ...]
            "steering": [], # [float, ...]
            "reward": [], # [array, ...]
            "dense_reward": [],
            "goal_position": {'x': goal_dict['pos'][0], 'y': goal_dict['pos'][1]},
            "goal_heading": goal_dict['heading'],
            "goal_speed": goal_dict['speed'],
            "width": veh.getWidth(),
            "length": veh.getLength(),
            "type": get_object_type_str(veh),
            "timestep": [],
            "planner_rtgs": [],
            "next_planner_acceleration": 0.,
            "next_planner_steering": 0.,
            "adversary_rtgs": [],
            "next_adversary_acceleration": 0.,
            "next_adversary_steering": 0.
        }


    def update_vehicle_data_dict(self, t, vehicles, vehicle_data_dict, goal_dict, goal_dist_normalizer, gt_data_dict, preproc_data, road_edge_polylines):
        for veh_idx, veh in enumerate(vehicles):
            veh_id = veh.getID()
            gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
            vehicle_data_dict[veh_id]["gt_position"].append({'x': gt_traj_data[t, 0], 'y': gt_traj_data[t, 1]})
            vehicle_data_dict[veh_id]["gt_heading"].append(gt_traj_data[t, 2])
            vehicle_data_dict[veh_id]["gt_speed"].append(gt_traj_data[t, 3])
            # can only compute central difference approximation to acceleration at intermediate timesteps
            if t > 0 and t < self.steps - 1:
                gt_accel = (gt_traj_data[t+1, 3] - gt_traj_data[t-1, 3]) / (2 * self.dt)
                vehicle_data_dict[veh_id]["gt_acceleration"].append(gt_accel)
            else:
                vehicle_data_dict[veh_id]["gt_acceleration"].append(0)
            
            vehicle_data_dict[veh_id]['position'].append({'x': veh.getPosition().x, 'y': veh.getPosition().y})
            vehicle_data_dict[veh_id]["velocity"].append({'x': veh.velocity().x, 'y': veh.velocity().y})
            vehicle_data_dict[veh_id]["heading"].append(veh.getHeading())
            vehicle_data_dict[veh_id]["timestep"].append(t)
            
            veh_exists = gt_traj_data[t, 4]
            if t > 0 and vehicle_data_dict[veh_id]["existence"][-1] == 0:
                veh_exists = 0
            vehicle_data_dict[veh_id]["existence"].append(veh_exists)
            
            if self.adversary.real_time_rewards:
                if t == 0:
                    unnormalized_rtg = preproc_data['rtgs'][veh_idx, t]
                    unnormalized_rtg = np.concatenate([unnormalized_rtg[:1], unnormalized_rtg[3:]], axis=-1)
                    # the maximum achievable return
                    unnormalized_rtg[0] = 10
                    unnormalized_rtg[1] = 90
                    unnormalized_rtg[2] = 90
                    
                    # the minimum achievable return
                    if veh_id == self.adversary_vehicle:
                        unnormalized_rtg[0] = 0
                        unnormalized_rtg[1] = -10
                        unnormalized_rtg[2] = -10
                    vehicle_data_dict[veh_id]["adversary_rtgs"].append(unnormalized_rtg)
                else:
                    # normalize the rtg in the forward pass of the ctrl_sim model
                    discounted_unnormalized_rtg = vehicle_data_dict[veh_id]["adversary_rtgs"][-1] - vehicle_data_dict[veh_id]["dense_reward"][-1]
                    vehicle_data_dict[veh_id]["adversary_rtgs"].append(discounted_unnormalized_rtg)
            
            reward = compute_reward(self.cfg.nocturne['rew_cfg'], veh, goal_dict[veh_id], goal_dist_normalizer[veh_id], vehicle_data_dict, collision_fix=self.cfg.nocturne.collision_fix)
            vehicle_data_dict[veh_id]["reward"].append(reward)
 
        if self.adversary.real_time_rewards:
            vehicle_data_dict = self.compute_dense_reward(t, vehicle_data_dict, road_edge_polylines)
        else:
            vehicle_data_dict = self.compute_nearest_dist_all(t, vehicle_data_dict)

        return vehicle_data_dict


    def apply_adv_traj(self, veh, t, gt_data_dict, vehicle_data_dict, adv_traj):
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
            bike_model = BicycleModel(x=adv_traj[t+1, 0],
                                      y=adv_traj[t+1, 1],
                                      theta=adv_traj[t+1, 4],
                                      vel=np.sqrt(adv_traj[t+1, 2] ** 2 + adv_traj[t+1, 3] ** 2),
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


    def update_running_statistics(self, data_dict):
        # ego metrics
        collisions_scenario = []
        collisions_w_adv_scenario = []
        offroads_scenario = []
        scenario_has_adv_ego_collision = False
        
        for v in [self.ego_vehicle]:
            ego_mask = np.array(data_dict[v]["existence"]).astype(bool)
            # only evaluate future timesteps
            future_mask = np.zeros(self.steps + 1).astype(bool)
            future_mask[self.history_steps:] = True
            ego_mask = ego_mask * future_mask

            if ego_mask.sum() != 0:
                # update goal success rate
                rew = np.array(data_dict[v]["reward"])[ego_mask]
                goal_achieved = np.any(np.sum(rew[:, :1], axis=1) == 1)
                self.goal_achieved_all.append(float(goal_achieved))

                # update collision statistics
                has_collision = float(np.any(rew[:, 6] == 1))
                collisions_scenario.append(has_collision)

                # update offroad statistics
                has_offroad = float(np.any(rew[:, 7] == 1))
                offroads_scenario.append(has_offroad)

            # we have self.steps transition steps and therefore self.steps+1 states
            gt_position_x = np.array([data_dict[v]["gt_position"][t]['x'] for t in range(self.steps + 1)])
            gt_position_y = np.array([data_dict[v]["gt_position"][t]['y'] for t in range(self.steps + 1)])
            sim_position_x = np.array([data_dict[v]["position"][t]['x'] for t in range(self.steps + 1)])
            sim_position_y = np.array([data_dict[v]["position"][t]['y'] for t in range(self.steps + 1)])

            simulated_positions = np.array([sim_position_x, sim_position_y]).transpose(1, 0)
            gt_positions = np.array([gt_position_x, gt_position_y]).transpose(1, 0)
            
            if ego_mask.sum() != 0:
                # update ade
                ade = np.linalg.norm(simulated_positions[ego_mask] - gt_positions[ego_mask], axis=1).mean()
                self.ades_all.append(ade)

                # update fde
                last_position = np.where(ego_mask == 1)[-1][-1]
                fde = np.linalg.norm(simulated_positions[last_position] - gt_positions[last_position])
                self.fdes_all.append(fde)

                if goal_achieved:
                    progress = np.linalg.norm(np.diff(simulated_positions[self.history_steps:last_position+1], axis=0), axis=-1).sum()
                else:
                    dist_to_goal = np.linalg.norm(simulated_positions[self.history_steps:last_position+1] - np.expand_dims(gt_positions[last_position], axis=0), axis=-1)
                    getting_closer = np.diff(dist_to_goal) < 0
                    per_timestep_dists = np.linalg.norm(np.diff(simulated_positions[self.history_steps:last_position+1], axis=0), axis=-1)
                    valid_per_timestep_dists = per_timestep_dists[getting_closer]
                    progress = valid_per_timestep_dists.sum()
                self.progress_all.append(progress)

                ego_accels = np.array(data_dict[v]["acceleration"])[ego_mask]
                ego_jerk = np.abs(np.diff(ego_accels)) / self.dt
                self.ego_jerk_all.append(ego_jerk)
                self.ego_accel_all.append(np.abs(ego_accels))

                ego_steering = np.array(data_dict[v]["steering"])[ego_mask]
                ego_steering_rate = np.abs(np.diff(ego_steering)) / self.dt 
                self.ego_steering_rate_all.append(ego_steering_rate)
        
        for v in [self.adversary_vehicle]:
            adv_mask = np.array(data_dict[v]["existence"]).astype(bool)
            # only evaluate future timesteps
            future_mask = np.zeros(self.steps + 1).astype(bool)
            future_mask[self.history_steps:] = True
            adv_mask = adv_mask * future_mask

            if adv_mask.sum() != 0:
                # update lin speed jsd statistics
                sim_velocity_x = np.array([data_dict[v]["velocity"][t]['x'] for t in range(self.steps + 1)])[adv_mask]
                sim_velocity_y = np.array([data_dict[v]["velocity"][t]['y'] for t in range(self.steps + 1)])[adv_mask]
                sim_velocities = np.array([sim_velocity_x, sim_velocity_y]).transpose(1, 0)

                sim_lin_speeds = np.linalg.norm(sim_velocities, axis=1)
                gt_lin_speeds = np.array(data_dict[v]["gt_speed"])[adv_mask]

                self.lin_speed_sim_all.append(sim_lin_speeds[:, None])
                self.lin_speed_gt_all.append(gt_lin_speeds[:, None])

                # update ang speed jsd statistics
                sim_ang_speeds = np.array(data_dict[v]["heading"])[adv_mask] / self.dt
                gt_ang_speeds = np.array(data_dict[v]["gt_heading"])[adv_mask] / self.dt

                self.ang_speed_sim_all.append(sim_ang_speeds[:, None])
                self.ang_speed_gt_all.append(gt_ang_speeds[:, None])

                # update accel jsd statistics
                gt_accels = np.array(data_dict[v]["gt_acceleration"])[adv_mask]
                sim_accels = np.array(data_dict[v]["acceleration"])[adv_mask]

                # we do not have gt acceleration for endpoints as we do central difference approximation to ground-truth acceleration
                accel_mask = np.ones(gt_accels.shape).astype(bool)
                accel_mask[0] = False
                accel_mask[-1] = False

                gt_accels = gt_accels[accel_mask]
                sim_accels = sim_accels[accel_mask]
                self.accel_sim_all.append(sim_accels[:, None])
                self.accel_gt_all.append(gt_accels[:, None])

                # update nearest dist jsd statistics
                gt_nearest_dists = np.array(data_dict[v]["gt_nearest_dist"])[adv_mask]
                sim_nearest_dists = np.array(data_dict[v]["nearest_dist"])[adv_mask]
                self.nearest_dist_gt_all.append(gt_nearest_dists[:, None])
                self.nearest_dist_sim_all.append(sim_nearest_dists[:, None])
        
        
        ego_mask = np.array(data_dict[self.ego_vehicle]["existence"]).astype(bool)
        adv_mask = np.array(data_dict[self.adversary_vehicle]["existence"]).astype(bool)
        # only evaluate future timesteps
        future_mask = np.zeros(self.steps + 1).astype(bool)
        future_mask[self.history_steps:] = True
        ego_mask = ego_mask * future_mask
        adv_mask = adv_mask * future_mask

        if ego_mask.sum() != 0 and adv_mask.sum() != 0:
            # update goal success rate
            ego_coll_rew = np.array(data_dict[self.ego_vehicle]["reward"])[ego_mask, 6]
            adv_coll_rew = np.array(data_dict[self.adversary_vehicle]["reward"])[adv_mask, 6]
            shortest_len = min(len(ego_coll_rew), len(adv_coll_rew))
            ego_coll_rew = ego_coll_rew[:shortest_len]
            adv_coll_rew = adv_coll_rew[:shortest_len]
            coll_with_adv_mask = ((ego_coll_rew == adv_coll_rew).astype(float) * ego_coll_rew).astype(bool)
            has_collision_w_adv = float(np.any(coll_with_adv_mask))
            # double check that collision is between ego and adversary by comparing distance at time of collision
            if has_collision_w_adv == 1.:
                coll_ids = np.where(coll_with_adv_mask)[0]
                ego_position_x = np.array([data_dict[self.ego_vehicle]["position"][t]['x'] for t in range(self.steps + 1)])
                ego_position_y = np.array([data_dict[self.ego_vehicle]["position"][t]['y'] for t in range(self.steps + 1)])
                ego_positions = np.array([ego_position_x, ego_position_y]).transpose(1, 0)[ego_mask]
                ego_positions = ego_positions[:shortest_len]
                adv_position_x = np.array([data_dict[self.adversary_vehicle]["position"][t]['x'] for t in range(self.steps + 1)])
                adv_position_y = np.array([data_dict[self.adversary_vehicle]["position"][t]['y'] for t in range(self.steps + 1)])
                adv_positions = np.array([adv_position_x, adv_position_y]).transpose(1, 0)[adv_mask]
                adv_positions = adv_positions[:shortest_len]

                adv_velocity_x = np.array([data_dict[self.adversary_vehicle]["velocity"][t]['x'] for t in range(self.steps + 1)])
                adv_velocity_y = np.array([data_dict[self.adversary_vehicle]["velocity"][t]['y'] for t in range(self.steps + 1)])

                valid_coll_with_adv = False
                for coll_id in coll_ids:
                    if np.linalg.norm(ego_positions[coll_id] - adv_positions[coll_id]) < data_dict[self.ego_vehicle]["length"] + data_dict[self.adversary_vehicle]["length"]:
                        valid_coll_with_adv = True 
                        self.collision_speed_with_ego.append(np.sqrt(adv_velocity_x[coll_id] ** 2 + adv_velocity_y ** 2))
                        break 

                if not valid_coll_with_adv:
                    has_collision_w_adv = 0.
            
            collisions_w_adv_scenario.append(has_collision_w_adv)
            if has_collision_w_adv:
                scenario_has_adv_ego_collision = True
        
        if len(collisions_scenario) > 0:
            self.collision_rate_scenario.append(np.array(collisions_scenario).mean())
            if len(collisions_w_adv_scenario) == 0:
                collisions_w_adv_scenario.append(0.)
            self.collision_rate_w_adv_scenario.append(np.array(collisions_w_adv_scenario).mean())
            self.offroad_rate_scenario.append(np.array(offroads_scenario).mean())

        return scenario_has_adv_ego_collision


    def compute_metrics(self):
        metrics_dict = {}

        metrics_dict['ego_goal'] = np.array(self.goal_achieved_all).mean()
        metrics_dict['ego_prog'] = np.array(self.progress_all).mean()
        metrics_dict['ego_cr'] = np.array(self.collision_rate_scenario).mean()
        metrics_dict['ego_cr_w_adv'] = np.array(self.collision_rate_w_adv_scenario).mean()
        metrics_dict['ego_or'] = np.array(self.offroad_rate_scenario).mean()
        metrics_dict['ego_fde'] = np.array(self.fdes_all).mean()
        metrics_dict['ego_ade'] = np.array(self.ades_all).mean()
        metrics_dict['ego_accel'] = np.concatenate(self.ego_accel_all, axis=0).mean()
        metrics_dict['ego_jerk'] = np.concatenate(self.ego_jerk_all, axis=0).mean()
        metrics_dict['ego_steer_rate'] = np.concatenate(self.ego_steering_rate_all, axis=0).mean()
        metrics_dict['adv_coll_speed'] = np.array(self.collision_speed_with_ego).mean()

        # lin speed jsd 
        lin_speeds_gt = np.concatenate(self.lin_speed_gt_all, axis=0)
        lin_speeds_sim = np.concatenate(self.lin_speed_sim_all, axis=0)
        lin_speeds_gt = np.clip(lin_speeds_gt, 0, 30)
        lin_speeds_sim = np.clip(lin_speeds_sim, 0, 30)
        bin_edges = np.arange(201) * 0.5 * (100 / 30)
        P_lin_speeds_sim = np.histogram(lin_speeds_sim, bins=bin_edges)[0] / len(lin_speeds_sim)
        Q_lin_speeds_sim = np.histogram(lin_speeds_gt, bins=bin_edges)[0] / len(lin_speeds_gt)
        metrics_dict['adv_lin_jsd'] = distance.jensenshannon(P_lin_speeds_sim, Q_lin_speeds_sim)
        
        # ang speed jsd
        ang_speeds_gt = np.concatenate(self.ang_speed_gt_all, axis=0)
        ang_speeds_sim = np.concatenate(self.ang_speed_sim_all, axis=0)
        ang_speeds_gt = np.clip(ang_speeds_gt, -50, 50)
        ang_speeds_sim = np.clip(ang_speeds_sim, -50, 50)
        bin_edges = np.arange(201) * 0.5 - 50 
        P_ang_speeds_sim = np.histogram(ang_speeds_sim, bins=bin_edges)[0] / len(ang_speeds_sim)
        Q_ang_speeds_sim = np.histogram(ang_speeds_gt, bins=bin_edges)[0] / len(ang_speeds_gt)
        metrics_dict['adv_ang_jsd'] = distance.jensenshannon(P_ang_speeds_sim, Q_ang_speeds_sim)

        # accel jsd
        # discretize then undiscretize gt actions
        accels_gt = np.concatenate(self.accel_gt_all, axis=0)
        accels_gt =  ((np.clip(accels_gt, a_min=self.cfg_rl_waymo.min_accel, a_max=self.cfg_rl_waymo.max_accel) - self.cfg_rl_waymo.min_accel)
                             / (self.cfg_rl_waymo.max_accel - self.cfg_rl_waymo.min_accel))
        accels_gt = np.round(accels_gt * (self.cfg_rl_waymo.accel_discretization - 1))
        accels_gt /= (self.cfg_rl_waymo.accel_discretization - 1)
        accels_gt = (accels_gt * (self.cfg_rl_waymo.max_accel - self.cfg_rl_waymo.min_accel)) + self.cfg_rl_waymo.min_accel
        accels_sim = np.concatenate(self.accel_sim_all, axis=0)
        bin_edges = np.arange(self.cfg_rl_waymo.accel_discretization + 1) * 2 - self.cfg_rl_waymo.accel_discretization
        P_accels_sim = np.histogram(accels_sim, bins=bin_edges)[0] / len(accels_sim)
        Q_accels_sim = np.histogram(accels_gt, bins=bin_edges)[0] / len(accels_gt)
        metrics_dict['adv_acc_jsd'] = distance.jensenshannon(P_accels_sim, Q_accels_sim)

        # nearest dist jsd
        nearest_dists_gt = np.concatenate(self.nearest_dist_gt_all, axis=0)
        nearest_dists_sim = np.concatenate(self.nearest_dist_sim_all, axis=0)
        nearest_dists_gt = np.clip(nearest_dists_gt, 0, 40)
        nearest_dists_sim = np.clip(nearest_dists_sim, 0, 40)
        bin_edges = np.arange(201) * 0.5 * (100 / 40)
        P_nearest_dists_sim = np.histogram(nearest_dists_sim, bins=bin_edges)[0] / len(nearest_dists_sim)
        Q_nearest_dists_sim = np.histogram(nearest_dists_gt, bins=bin_edges)[0] / len(nearest_dists_gt)
        metrics_dict['nearest_dist_jsd'] = distance.jensenshannon(P_nearest_dists_sim, Q_nearest_dists_sim)
        
        return metrics_dict, ["{}: {:.6f}".format(k,v) for (k,v) in metrics_dict.items()]


    def get_planner_adversary(self, gt_data_dict, file):
        def get_veh_id(position_to_match):
            matched_veh_id = None
            for veh_id in gt_data_dict.keys():
                gt_traj = np.array(gt_data_dict[veh_id]["traj"])
                if np.all(np.isclose(gt_traj[0, :2], position_to_match, atol=1e-6)):
                    matched_veh_id = veh_id 
                    break 
            
            return matched_veh_id

        planner_dict_idx = self.file_to_id[self.test_filenames[file]]
        ego_id = self.eval_planner_dict[planner_dict_idx]['nocturne_sdc_id']
        adversary_id = self.eval_planner_dict[planner_dict_idx]['nocturne_adversary_id']

        # not valid without an adversary trajectory from CAT (need to evaluate over the same scenes even if we don't use CAT)
        if 'adv_traj' not in self.eval_planner_dict[planner_dict_idx]:
            return None, None, None
                
        nocturne_waymo_json_file_path = os.path.join(self.cfg.nocturne_waymo_val_interactive_folder, self.test_filenames[file])
        with open(nocturne_waymo_json_file_path, 'r') as f:
            json_dict = json.load(f)

        ego_veh_id = get_veh_id(np.array([json_dict['objects'][ego_id]['position'][0]['x'], json_dict['objects'][ego_id]['position'][0]['y']]))
        adversary_veh_id = get_veh_id(np.array([json_dict['objects'][adversary_id]['position'][0]['x'], json_dict['objects'][adversary_id]['position'][0]['y']]))
        
        adv_pos = self.eval_planner_dict[planner_dict_idx]['adv_traj']
        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
        adv_vel = get_polyline_vel(adv_pos)
        adv_traj = np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1)
        
        return ego_veh_id, adversary_veh_id, adv_traj


    def evaluate_planner_adversary(self):
        self.reset()
        
        if self.cfg.eval_planner_adversary.visualize:
            output_dir = cfg.eval_planner_adversary.viz_data_files
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        files = list(np.arange(len(self.test_filenames)))
        num_files_evaluated = 0
        print(f"Evaluating {self.planner.name} planner ({self.planner.model_path}) using seed={self.cfg.eval_planner_adversary.seed} against {self.adversary.name} adversary. ({self.adversary.model_path})")
        for enum, file in tqdm(enumerate(files)):
            if num_files_evaluated == self.cfg.eval_planner_adversary.num_files_to_evaluate:
                break
            
            gt_data_dict = get_ground_truth_states(self.cfg, self.cfg.nocturne_waymo_val_interactive_folder, self.test_filenames, file, self.dt, self.steps)
            sim, scenario, vehicles = self.load_scenario(self.cfg.nocturne_waymo_val_interactive_folder, file)
            # Allows us to quickly retrieve map features and initial RTG
            preproc_data, file_exists = self.load_preprocessed_data(os.path.join(self.cfg.dataset_root, 'preprocess/val_interactive'), file)

            if not file_exists:
                continue
            
            # adv_traj is the DenseTNT generated adversarial trajectory
            ego_veh_id, adversary_veh_id, adv_traj = self.get_planner_adversary(gt_data_dict, file)
            
            if ego_veh_id is None or adversary_veh_id is None:
                continue
            
            self.ego_vehicle = ego_veh_id 
            self.adversary_vehicle = adversary_veh_id

            num_files_evaluated += 1
            road_data = get_road_data(scenario)
            road_edge_polylines = self.extract_road_edge_polylines(road_data)
            vehicle_data_dict = {}
            goal_dict = {}
            goal_dist_normalizer = {}

            # initialize vehicle_data_dict
            for veh in vehicles:
                veh_id = veh.getID()
                gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
                goal_dict[veh_id] = self.initialize_goal_dict(veh, gt_traj_data)
                vehicle_data_dict[veh_id] = self.initialize_vehicle_data_dict(veh, goal_dict[veh_id])
                goal_dist_normalizer[veh_id] = self.compute_goal_dist_normalizer(veh, goal_dict[veh_id]['pos'])
            
            self.planner.reset(vehicle_data_dict)
            self.adversary.reset(vehicle_data_dict)
            
            for t in range(self.steps):
                
                vehicle_data_dict = self.update_vehicle_data_dict(t, 
                                                                  vehicles, 
                                                                  vehicle_data_dict, 
                                                                  goal_dict,
                                                                  goal_dist_normalizer,
                                                                  gt_data_dict, 
                                                                  preproc_data,
                                                                  road_edge_polylines)

                self.planner.update_state(vehicle_data_dict, [self.ego_vehicle], t)
                if not self.adversary.name == 'cat':
                    self.adversary.update_state(vehicle_data_dict, [self.adversary_vehicle], t)
                vehicle_data_dict = self.planner.predict(vehicle_data_dict, gt_data_dict, preproc_data, self.preprocessed_dset, [self.ego_vehicle], t)
                if not self.adversary.name == 'cat':
                    vehicle_data_dict = self.adversary.predict(vehicle_data_dict, gt_data_dict, preproc_data, self.preprocessed_dset, [self.adversary_vehicle], t)
                
                for veh in vehicles:
                    veh_id = veh.getID()
                    
                    if t >= self.history_steps - 1 and veh_id == self.ego_vehicle:
                        veh, veh_action = self.planner.act(veh, t, vehicle_data_dict)
                    elif t >= self.history_steps - 1 and veh_id == self.adversary_vehicle:
                        # apply hardcoded cat trajectory (non-reactive)
                        if self.adversary.name == 'cat':
                            veh, veh_action = self.apply_adv_traj(veh, t, gt_data_dict, vehicle_data_dict, adv_traj)
                        else:
                            veh, veh_action = self.adversary.act(veh, t, vehicle_data_dict)
                            
                    else:
                        veh, veh_action = self.apply_gt_action(veh, t, gt_data_dict, vehicle_data_dict)
                    vehicle_data_dict[veh_id]["acceleration"].append(veh_action[0])
                    vehicle_data_dict[veh_id]["steering"].append(veh_action[1])

                sim.step(self.dt)

            vehicle_data_dict = self.update_vehicle_data_dict(self.steps, 
                                                            vehicles, 
                                                            vehicle_data_dict, 
                                                            goal_dict,
                                                            goal_dist_normalizer,
                                                            gt_data_dict, 
                                                            preproc_data,
                                                            road_edge_polylines)
            
            for veh in vehicles:
                veh_id = veh.getID()
                vehicle_data_dict[veh_id]["acceleration"].append(0)
                vehicle_data_dict[veh_id]["steering"].append(0)
            
            scenario_has_adv_ego_collision = self.update_running_statistics(vehicle_data_dict)

            if self.cfg.eval_planner_adversary.visualize:
                if scenario_has_adv_ego_collision:
                    file_name = f"{self.test_filenames[file].split('.')[0]}_success.json"
                else:
                    file_name = f"{self.test_filenames[file].split('.')[0]}_fail.json"
                print(f"Saving scenario: {file_name}")
                output_vehicle_data_dict = {}
                attributes = ['position', 'velocity', 'heading', 'existence', 'acceleration', 'steering', 'reward', 'goal_position', 'width', 'length', 'type']
                for veh_id in vehicle_data_dict.keys():
                    output_vehicle_data_dict[veh_id] = {}
                    for a in attributes:
                        output_vehicle_data_dict[veh_id][a] = vehicle_data_dict[veh_id][a]
                
                road_data = get_road_data(scenario)
                veh_ids = [v.getID() for v in vehicles]
                planner_idx = veh_ids.index(self.ego_vehicle)
                adversary_idx = veh_ids.index(self.adversary_vehicle)
                export_data = {"name": file_name, "objects": [*output_vehicle_data_dict.values()], "roads": road_data, 'planner': planner_idx, 'adversary': adversary_idx}

                with open(os.path.join(output_dir, file_name), 'w') as f:
                    json.dump(export_data, f)
        
            if self.cfg.eval_planner_adversary.verbose:
                print(self.compute_metrics()[-1])
        
        return self.compute_metrics()