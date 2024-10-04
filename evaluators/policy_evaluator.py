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
from utils.data import *
from utils.viz import *
from datasets.rl_waymo import RLWaymoDatasetCtRLSim, RLWaymoDatasetCTGPlusPlus
from nocturne.bicycle_model import BicycleModel
from evaluators.evaluator import Evaluator
from policies import AutoregressivePolicy, CTGPlusPlusPolicy

class PolicyEvaluator(Evaluator):
    def __init__(self, cfg, policy):
        super(PolicyEvaluator, self).__init__(cfg)

        self.policy = policy

        with open(os.path.join(cfg.dataset_root, 'test_filenames.pkl'), 'rb') as f:
            test_filenames_dict = pickle.load(f)
        self.test_filenames = test_filenames_dict['test_filenames']
        self.vehicles_to_evaluate = None
        self.is_ctg_plus_plus = isinstance(self.policy, CTGPlusPlusPolicy)
        if self.is_ctg_plus_plus:
            self.preprocessed_dset = RLWaymoDatasetCTGPlusPlus(self.policy.cfg, split_name='test', mode='eval')
        else:
            self.preprocessed_dset = RLWaymoDatasetCtRLSim(self.policy.cfg, split_name='test', mode='eval')
        self.reset()

    
    def reset(self):
        seed = self.cfg.eval.seed
        random.seed(seed)  # Python random module.
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  # PyTorch.
        torch.cuda.manual_seed(seed)  # PyTorch, for CUDA.
        
        # reconstruction metrics
        self.ades_all = []
        self.fdes_all = []
        self.goal_achieved_all = []
        # distributional realism metrics
        self.lin_speed_sim_all = []
        self.lin_speed_gt_all = []
        self.ang_speed_sim_all = []
        self.ang_speed_gt_all = []
        self.accel_sim_all = []
        self.accel_gt_all = []
        self.nearest_dist_sim_all = []
        self.nearest_dist_gt_all = []
        # common sense metrics 
        self.collision_rate_scenario = []
        self.offroad_rate_scenario = []


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
            "rtgs": [],
            "next_acceleration": 0.,
            "next_steering": 0.
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
            
            if self.policy.real_time_rewards:
                if t == 0:
                    unnormalized_rtg = preproc_data['rtgs'][veh_idx, t]
                    unnormalized_rtg = np.concatenate([unnormalized_rtg[:1], unnormalized_rtg[3:]], axis=-1)
                    if self.policy.max_return:
                        # the maximum achievable return
                        unnormalized_rtg[0] = 10
                        unnormalized_rtg[1] = 90
                        unnormalized_rtg[2] = 90
                    
                    elif self.policy.min_return:
                        # the maximum achievable return
                        unnormalized_rtg[0] = 10
                        unnormalized_rtg[1] = 90
                        unnormalized_rtg[2] = 90
                    
                        # for evaluated vehicles, set to the minimum possible return
                        if veh_id in self.vehicles_to_evaluate:
                            unnormalized_rtg[0] = 0
                            unnormalized_rtg[1] = -10
                            unnormalized_rtg[2] = -10

                    vehicle_data_dict[veh_id]["rtgs"].append(unnormalized_rtg)
                else:
                    # normalize the rtg in the forward pass of the ctrl_sim model
                    discounted_unnormalized_rtg = vehicle_data_dict[veh_id]["rtgs"][-1] - vehicle_data_dict[veh_id]["dense_reward"][-1]
                    vehicle_data_dict[veh_id]["rtgs"].append(discounted_unnormalized_rtg)
            
            reward = compute_reward(self.cfg.nocturne['rew_cfg'], veh, goal_dict[veh_id], goal_dist_normalizer[veh_id], vehicle_data_dict, collision_fix=self.cfg.nocturne.collision_fix)
            vehicle_data_dict[veh_id]["reward"].append(reward)
 
        if self.policy.real_time_rewards:
            vehicle_data_dict = self.compute_dense_reward(t, vehicle_data_dict, road_edge_polylines)
        else:
            vehicle_data_dict = self.compute_nearest_dist_all(t, vehicle_data_dict)

        return vehicle_data_dict


    def update_running_statistics(self, data_dict):
        veh_ids = self.vehicles_to_evaluate
        
        collisions_scenario = []
        offroads_scenario = []
        for v in veh_ids:
            mask = np.array(data_dict[v]["existence"]).astype(bool)
            # only evaluate future timesteps
            future_mask = np.zeros(self.steps + 1).astype(bool)
            future_mask[self.history_steps:] = True
            mask = mask * future_mask

            if mask.sum() != 0:
                # update goal success rate
                rew = np.array(data_dict[v]["reward"])[mask]
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
            
            if mask.sum() != 0:
                # update ade
                ade = np.linalg.norm(simulated_positions[mask] - gt_positions[mask], axis=1).mean()
                self.ades_all.append(ade)

                # update fde
                last_position = np.where(mask == 1)[-1][-1]
                fde = np.linalg.norm(simulated_positions[last_position] - gt_positions[last_position])
                self.fdes_all.append(fde)

                # update lin speed jsd statistics
                sim_velocity_x = np.array([data_dict[v]["velocity"][t]['x'] for t in range(self.steps + 1)])[mask]
                sim_velocity_y = np.array([data_dict[v]["velocity"][t]['y'] for t in range(self.steps + 1)])[mask]
                sim_velocities = np.array([sim_velocity_x, sim_velocity_y]).transpose(1, 0)

                sim_lin_speeds = np.linalg.norm(sim_velocities, axis=1)
                gt_lin_speeds = np.array(data_dict[v]["gt_speed"])[mask]

                self.lin_speed_sim_all.append(sim_lin_speeds[:, None])
                self.lin_speed_gt_all.append(gt_lin_speeds[:, None])

                # update ang speed jsd statistics
                sim_ang_speeds = np.array(data_dict[v]["heading"])[mask] / self.dt
                gt_ang_speeds = np.array(data_dict[v]["gt_heading"])[mask] / self.dt

                self.ang_speed_sim_all.append(sim_ang_speeds[:, None])
                self.ang_speed_gt_all.append(gt_ang_speeds[:, None])

                # update accel jsd statistics
                gt_accels = np.array(data_dict[v]["gt_acceleration"])[mask]
                sim_accels = np.array(data_dict[v]["acceleration"])[mask]

                # we do not have gt acceleration for endpoints as we do central difference approximation to ground-truth acceleration
                accel_mask = np.ones(gt_accels.shape).astype(bool)
                accel_mask[0] = False
                accel_mask[-1] = False

                gt_accels = gt_accels[accel_mask]
                sim_accels = sim_accels[accel_mask]
                self.accel_sim_all.append(sim_accels[:, None])
                self.accel_gt_all.append(gt_accels[:, None])

                # update nearest dist jsd statistics
                gt_nearest_dists = np.array(data_dict[v]["gt_nearest_dist"])[mask]
                sim_nearest_dists = np.array(data_dict[v]["nearest_dist"])[mask]
                self.nearest_dist_gt_all.append(gt_nearest_dists[:, None])
                self.nearest_dist_sim_all.append(sim_nearest_dists[:, None])

        
        if len(collisions_scenario) > 0:
            self.collision_rate_scenario.append(np.array(collisions_scenario).mean())
            self.offroad_rate_scenario.append(np.array(offroads_scenario).mean())


    def compute_metrics(self):
        metrics_dict = {}

        metrics_dict['goal'] = np.array(self.goal_achieved_all).mean()
        metrics_dict['collision_rate'] = np.array(self.collision_rate_scenario).mean()
        metrics_dict['offroad_rate'] = np.array(self.offroad_rate_scenario).mean()
        
        metrics_dict['fde'] = np.array(self.fdes_all).mean()
        metrics_dict['ade'] = np.array(self.ades_all).mean()

        # lin speed jsd 
        lin_speeds_gt = np.concatenate(self.lin_speed_gt_all, axis=0)
        lin_speeds_sim = np.concatenate(self.lin_speed_sim_all, axis=0)
        lin_speeds_gt = np.clip(lin_speeds_gt, 0, 30)
        lin_speeds_sim = np.clip(lin_speeds_sim, 0, 30)
        bin_edges = np.arange(201) * 0.5 * (100 / 30)
        P_lin_speeds_sim = np.histogram(lin_speeds_sim, bins=bin_edges)[0] / len(lin_speeds_sim)
        Q_lin_speeds_sim = np.histogram(lin_speeds_gt, bins=bin_edges)[0] / len(lin_speeds_gt)
        metrics_dict['lin_speed_jsd'] = distance.jensenshannon(P_lin_speeds_sim, Q_lin_speeds_sim)
        
        # ang speed jsd
        ang_speeds_gt = np.concatenate(self.ang_speed_gt_all, axis=0)
        ang_speeds_sim = np.concatenate(self.ang_speed_sim_all, axis=0)
        ang_speeds_gt = np.clip(ang_speeds_gt, -50, 50)
        ang_speeds_sim = np.clip(ang_speeds_sim, -50, 50)
        bin_edges = np.arange(201) * 0.5 - 50 
        P_ang_speeds_sim = np.histogram(ang_speeds_sim, bins=bin_edges)[0] / len(ang_speeds_sim)
        Q_ang_speeds_sim = np.histogram(ang_speeds_gt, bins=bin_edges)[0] / len(ang_speeds_gt)
        metrics_dict['ang_speed_jsd'] = distance.jensenshannon(P_ang_speeds_sim, Q_ang_speeds_sim)

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
        metrics_dict['accel_jsd'] = distance.jensenshannon(P_accels_sim, Q_accels_sim)

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


    def find_interesting_agent(self, vehicles, gt_data_dict):
        # first, we extract the goal positions of the agents that
        # (a)  are close to another moving agent (filter for agent goals within 10 metres)
        # (b)  timestep of goal is within 2 seconds (20 timesteps) of close vehicle
        # (c)  each trajectory has length at least 60
        # Then, randomly sample an interesting agent from this set
        goals = []
        goal_timesteps = []
        has_thirty_timesteps = []
        veh_ids = []
        for veh in vehicles:
            veh_id = veh.getID()
            # must be a moving vehicle
            if veh_id not in self.vehicles_to_evaluate:
                continue
            
            goal_pos = np.array([veh.target_position.x, veh.target_position.y])
            idx_goal = self.steps - 1
            gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj_data[:, 4]
            idx_disappear = np.where(existence_mask == 0)[0]
            if len(idx_disappear) > 0:
                idx_goal = idx_disappear[0] - 1
                if np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
                    goal_pos = gt_traj_data[idx_goal, :2]
            
            goal_timesteps.append(idx_goal - self.history_steps)
            goals.append(goal_pos)
            has_thirty_timesteps.append(1 if existence_mask[self.history_steps:].sum() >= self.cfg.eval.interesting_traj_len_threshold else 0)
            veh_ids.append(veh_id)
        
        goals = np.array(goals)
        goal_timesteps = np.array(goal_timesteps)
        has_thirty_timesteps = np.array(has_thirty_timesteps)
        dists = np.linalg.norm(np.expand_dims(goals, 0) - np.expand_dims(goals, 1), 2, -1)
        nearby_goal_mask = dists < self.cfg.eval.interesting_goal_dist_threshold
        not_same_goal_mask = dists > 0 
        has_thirty_timesteps_mask = np.repeat(has_thirty_timesteps[:, np.newaxis], has_thirty_timesteps.shape[0], 1)
        goal_timestep_difference = np.abs(goal_timesteps[:, np.newaxis] - goal_timesteps[np.newaxis, :])
        within_two_seconds_mask = goal_timestep_difference < self.cfg.eval.interesting_timestep_diff_threshold
        goal_mask = nearby_goal_mask * not_same_goal_mask * has_thirty_timesteps_mask * has_thirty_timesteps_mask.T * within_two_seconds_mask

        indices = np.where(goal_mask == 1)
        valid_pairs = list(zip(indices[0], indices[1]))
        if len(valid_pairs) > 0:
            samp = random.choice(valid_pairs)
            interesting_agent_id = samp[0]
            interesting_agent_veh_id = veh_ids[interesting_agent_id]
        else:
            interesting_agent_veh_id = None 
        
        return interesting_agent_veh_id


    def find_interesting_pair(self, vehicles, gt_data_dict):
        # first, we extract the goal positions of the agents that
        # (a)  are close to another moving agent (filter for agent goals within 10 metres)
        # (b)  timestep of goal is within 2 seconds (20 timesteps) of close vehicle
        # (c)  each trajectory has length at least 60
        # Then, randomly sample an interesting agent from this set
        goals = []
        goal_timesteps = []
        has_thirty_timesteps = []
        veh_ids = []
        for veh in vehicles:
            veh_id = veh.getID()
            # must be a moving vehicle
            if veh_id not in self.vehicles_to_evaluate:
                continue
            
            goal_pos = np.array([veh.target_position.x, veh.target_position.y])
            idx_goal = self.steps - 1
            gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj_data[:, 4]
            idx_disappear = np.where(existence_mask == 0)[0]
            if len(idx_disappear) > 0:
                idx_goal = idx_disappear[0] - 1
                if np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
                    goal_pos = gt_traj_data[idx_goal, :2]
            
            goal_timesteps.append(idx_goal - self.history_steps)
            goals.append(goal_pos)
            has_thirty_timesteps.append(1 if existence_mask[self.history_steps:].sum() >= self.cfg.eval.interesting_traj_len_threshold else 0)
            veh_ids.append(veh_id)
        
        goals = np.array(goals)
        goal_timesteps = np.array(goal_timesteps)
        has_thirty_timesteps = np.array(has_thirty_timesteps)
        dists = np.linalg.norm(np.expand_dims(goals, 0) - np.expand_dims(goals, 1), 2, -1)
        nearby_goal_mask = dists < self.cfg.eval.interesting_goal_dist_threshold
        not_same_goal_mask = dists > 0 
        has_thirty_timesteps_mask = np.repeat(has_thirty_timesteps[:, np.newaxis], has_thirty_timesteps.shape[0], 1)
        goal_timestep_difference = np.abs(goal_timesteps[:, np.newaxis] - goal_timesteps[np.newaxis, :])
        within_two_seconds_mask = goal_timestep_difference < self.cfg.eval.interesting_timestep_diff_threshold
        goal_mask = nearby_goal_mask * not_same_goal_mask * has_thirty_timesteps_mask * has_thirty_timesteps_mask.T * within_two_seconds_mask

        indices = np.where(goal_mask == 1)
        valid_pairs = list(zip(indices[0], indices[1]))
        if len(valid_pairs) > 0:
            samp = random.choice(valid_pairs)
            interesting_agent_id_1 = samp[0]
            interesting_agent_id_2 = samp[1]
            interesting_agent_pair = [veh_ids[interesting_agent_id_1], veh_ids[interesting_agent_id_2]]
        else:
            interesting_agent_pair = None 
        
        return interesting_agent_pair


    def print_header(self):
        to_print = f"Evaluating {self.policy.model_path} using seed={self.cfg.eval.seed} in {self.cfg.eval.eval_mode}"
        # if tilting is supported
        if self.policy.tilt_dict['tilt']:
            to_print = to_print + f" with tilting: [veh-veh={self.policy.veh_veh_tilt}, veh-edge={self.policy.veh_edge_tilt}, goal={self.policy.goal_tilt}]"
        if isinstance(self.policy, AutoregressivePolicy):
            to_print = to_print + f", action-temp={self.policy.action_temperature}"

    
    def evaluate_policy(self):
        self.reset()
        
        files = list(np.arange(len(self.test_filenames)))
        num_files_evaluated = 0
        if self.is_ctg_plus_plus:
            num_loops = -1
        
        self.print_header()
        for enum, file in tqdm(enumerate(files)):
            if num_files_evaluated == self.cfg.eval.num_files_to_evaluate // self.cfg.eval.partitions:
                break
            
            gt_data_dict = get_ground_truth_states(self.cfg, self.cfg.nocturne_waymo_val_folder, self.test_filenames, file, self.dt, self.steps)
            sim, scenario, vehicles = self.load_scenario(self.cfg.nocturne_waymo_val_folder, file)
            self.vehicles_to_evaluate = get_moving_vehicles(scenario)
            # Allows us to quickly retrieve map features and initial RTG
            preproc_data, file_exists = self.load_preprocessed_data(os.path.join(self.cfg.dataset_root, 'preprocess/test'), file)

            if not file_exists:
                continue
            
            # NOTE: As this is the only use of random, this ensures that different models (if seeded consistently)
            # are evaluated on the same set of agents.
            if self.cfg.eval.eval_mode == 'multi_agent':
                if len(self.vehicles_to_evaluate) > self.cfg.eval.multi_agent_eval_threshold:
                    veh_ids_to_evaluate = random.sample(self.vehicles_to_evaluate, self.cfg.eval.multi_agent_eval_threshold)
                else:
                    veh_ids_to_evaluate = self.vehicles_to_evaluate
            elif self.cfg.eval.eval_mode == 'one_agent':
                veh_id_to_evaluate = self.find_interesting_agent(vehicles, gt_data_dict.copy())
                veh_ids_to_evaluate = [veh_id_to_evaluate] if veh_id_to_evaluate is not None else None
            else:
                veh_ids_to_evaluate = self.find_interesting_pair(vehicles, gt_data_dict.copy())
            
            if veh_ids_to_evaluate:
                self.vehicles_to_evaluate = veh_ids_to_evaluate
            else:
                continue

            if self.is_ctg_plus_plus:
                num_loops += 1
                if self.cfg.eval.partition == 0:
                    if 0 <= num_loops < (self.cfg.eval.num_files_to_evaluate // self.cfg.eval.partitions):
                        pass
                    else:
                        continue
                elif self.cfg.eval.partition == 1:
                    if (self.cfg.eval.num_files_to_evaluate  // self.cfg.eval.partitions) <= num_loops < (2 * self.cfg.eval.num_files_to_evaluate // self.cfg.eval.partitions):
                        pass
                    else:
                        continue
                elif self.cfg.eval.partition == 2:
                    if (2 * self.cfg.eval.num_files_to_evaluate  // self.cfg.eval.partitions) <= num_loops < (3 * self.cfg.eval.num_files_to_evaluate // self.cfg.eval.partitions):
                        pass
                    else:
                        continue
                elif self.cfg.eval.partition == 3:
                    if (3 * self.cfg.eval.num_files_to_evaluate  // self.cfg.eval.partitions) <= num_loops:
                        pass
                    else:
                        continue
                else:
                    print("Code assumes 4 partitions, and not more.")
                    raise NotImplementedError

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
            
            self.policy.reset(vehicle_data_dict)
            if self.is_ctg_plus_plus:
                gt_data_dict_ctg_plus_plus = get_ground_truth_states(self.cfg, self.cfg.nocturne_waymo_val_folder, self.test_filenames, file, self.dt, self.steps, True)
                self.policy.update_gt_state(gt_data_dict_ctg_plus_plus)
                sampling_steps = np.array(list(range(0, self.steps+1, int((1 / self.policy.sampling_frequency) / self.dt)))) - 1
                sampling_steps = [i for i in sampling_steps if i >= self.history_steps - 1]

            for t in range(self.steps):
                vehicle_data_dict = self.update_vehicle_data_dict(t, 
                                                                  vehicles, 
                                                                  vehicle_data_dict, 
                                                                  goal_dict,
                                                                  goal_dist_normalizer,
                                                                  gt_data_dict, 
                                                                  preproc_data,
                                                                  road_edge_polylines)
                
                self.policy.update_state(vehicle_data_dict, self.vehicles_to_evaluate, t)
                if self.is_ctg_plus_plus:
                    if t in sampling_steps:
                        vehicle_data_dict = self.policy.predict(vehicle_data_dict, gt_data_dict, preproc_data, self.preprocessed_dset, self.vehicles_to_evaluate, t)
                else:
                    vehicle_data_dict = self.policy.predict(vehicle_data_dict, gt_data_dict, preproc_data, self.preprocessed_dset, self.vehicles_to_evaluate, t)
                
                for veh in vehicles:
                    veh_id = veh.getID()
                    
                    if t >= self.history_steps - 1 and veh_id in self.vehicles_to_evaluate:
                        veh, veh_action = self.policy.act(veh, t, vehicle_data_dict)
                    # uncontrolled agents are set to log replay through physics
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
            self.update_running_statistics(vehicle_data_dict)
            
            if self.cfg.eval.visualize:
                dir_name = f"viz_{self.test_filenames[file].split('.')[0]}"
                print(f"Saving scenario: {dir_name}")
                output_vehicle_data_dict = {}
                attributes = ['position', 'velocity', 'heading', 'existence', 'acceleration', 'steering', 'reward', 'goal_position', 'width', 'length', 'type']
                for veh_id in vehicle_data_dict.keys():
                    output_vehicle_data_dict[veh_id] = {}
                    for a in attributes:
                        output_vehicle_data_dict[veh_id][a] = vehicle_data_dict[veh_id][a]
                
                road_data = get_road_data(scenario)
                veh_ids = [v.getID() for v in vehicles]
                export_data = {"name": dir_name, "objects": [*output_vehicle_data_dict.values()], "roads": road_data}

                generate_video(export_data, str(enum), self.cfg.eval.movie_path)
            
            if self.cfg.eval.verbose:
                print(self.compute_metrics()[-1])

        if self.is_ctg_plus_plus:
            saved_scene_metrics = {"goal_success": self.goal_achieved_all, "ade": self.ades_all, "fde": self.fdes_all,
                                    "accel_gt": self.accel_gt_all, "accel_sim": self.accel_sim_all, 
                                    "ang_speed_gt": self.ang_speed_gt_all, "ang_speed_sim": self.ang_speed_sim_all, 
                                    "lin_speed_gt": self.lin_speed_gt_all, "lin_speed_sim": self.lin_speed_sim_all,
                                    "nearest_dist_gt": self.nearest_dist_gt_all, "nearest_dist_sim": self.nearest_dist_sim_all,
                                    "collision": self.collision_rate_scenario, "off_road": self.offroad_rate_scenario}
            
            output_dir = os.path.join("/".join(self.policy.model_path.split("/")[:-1]), "scene_results")
            os.makedirs(output_dir, exist_ok=True)
            output_filename =  os.path.join(output_dir, f"partition_{self.cfg.eval.partition}.json")

            json_serializable_data = convert_ndarray_to_list(saved_scene_metrics)       

            with open(output_filename, 'w') as f:
                json.dump(json_serializable_data, f)
        
        return self.compute_metrics()

