from models.ctg_plus_plus import CTGPlusPlus
from utils.data import get_object_type_onehot, add_batch_dim, from_numpy, MotionData
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import time
from policies.policy import Policy

class CTGPlusPlusPolicy(Policy):
    def __init__(self, 
                 cfg, 
                 model_path,
                 model,
                 use_rtg, 
                 predict_rtgs, 
                 discretize_rtgs, 
                 real_time_rewards, 
                 privileged_return, 
                 max_return,
                 min_return,
                 key_dict, 
                 tilt_dict, 
                 name,
                 sampling_frequency,
                 history_steps):

        super(CTGPlusPlusPolicy, self).__init__(cfg, 
                                                model_path,
                                                model,
                                                use_rtg, 
                                                predict_rtgs, 
                                                discretize_rtgs, 
                                                real_time_rewards, 
                                                privileged_return, 
                                                max_return,
                                                min_return,
                                                key_dict, 
                                                tilt_dict, 
                                                name)
        self.sampling_frequency = sampling_frequency
        self.history_steps = history_steps


    def get_data(self, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t):
        moving_ids = np.where(np.linalg.norm(self.states[:, 0, :2] - self.goals[:, 0, :2], axis=1) > self.cfg_rl_waymo.moving_threshold)[0]
        moving_agent_mask = np.isin(np.arange(self.states.shape[0]), moving_ids)
        
        # retrieve 10 most recent timesteps. By design, will always have at least 10 timesteps.
        ag_states = self.states[:, t-(self.history_steps - 1):t+1].copy()
        ag_states_future = self.gt_states[:, t+1: t+1+self.cfg_rl_waymo.train_context_length-self.history_steps].copy()
        ag_states = np.concatenate([ag_states, ag_states_future], axis=1)

        current_num_timesteps = ag_states.shape[1]
        if current_num_timesteps < self.cfg_rl_waymo.train_context_length:
            # states
            padded_agent_states = np.zeros((len(ag_states), self.cfg_rl_waymo.train_context_length, ag_states.shape[-1]))
            padded_agent_states[:, :current_num_timesteps] = ag_states
            ag_states = padded_agent_states.copy()

        ag_types = self.types.copy()
        if t == self.history_steps - 1:
            _actions = self.actions[:, t-(self.history_steps - 1):t].copy()
            actions = np.concatenate((np.zeros((len(_actions), 1, _actions.shape[-1])), _actions), axis=1)
        else:
            actions = self.actions[:, t-(self.history_steps):t].copy()
        rtgs = self.rtgs[:, t-(self.history_steps - 1):t+1].copy()
        goals = self.goals[:, t-(self.history_steps - 1):t+1].copy()
        timesteps = np.arange(self.cfg_rl_waymo.train_context_length)[:, None] + t-(self.history_steps - 1) 
        # always normalize to present timestep
        normalize_timestep = -1
        rtgs[:, :, 0] = ((np.clip(rtgs[:, :, 0], a_min=self.cfg_rl_waymo.min_rtg_pos, a_max=self.cfg_rl_waymo.max_rtg_pos) - self.cfg_rl_waymo.min_rtg_pos)
                             / (self.cfg_rl_waymo.max_rtg_pos - self.cfg_rl_waymo.min_rtg_pos))
        rtgs[:, :, 1] = ((np.clip(rtgs[:, :, 1], a_min=self.cfg_rl_waymo.min_rtg_veh, a_max=self.cfg_rl_waymo.max_rtg_veh) - self.cfg_rl_waymo.min_rtg_veh)
                             / (self.cfg_rl_waymo.max_rtg_veh - self.cfg_rl_waymo.min_rtg_veh))
        rtgs[:, :, 2] = ((np.clip(rtgs[:, :, 2], a_min=self.cfg_rl_waymo.min_rtg_road, a_max=self.cfg_rl_waymo.max_rtg_road) - self.cfg_rl_waymo.min_rtg_road)
                             / (self.cfg_rl_waymo.max_rtg_road - self.cfg_rl_waymo.min_rtg_road))
        
        motion_datas = {}
        new_agent_idx_dicts = {}
        dead_agent_veh_ids = []
        # veh_ids to evaluate in each motion_data, indexed by focal_id
        data_veh_ids = {}
        # which vehicle ids still have not been included in a data dictionary for processing
        unaccounted_veh_ids = vehicles_to_evaluate.copy()
        
        # sort in decreasing order by length
        lengths = []
        for veh_id in unaccounted_veh_ids:
            length = int(np.array(gt_data_dict[veh_id]['traj'])[:, 4].sum())
            lengths.append(length)
        sorted_idxs = np.argsort(np.array(lengths))[::-1]
        unaccounted_veh_ids = list(np.array(unaccounted_veh_ids)[sorted_idxs])
        
        while len(unaccounted_veh_ids) > 0:
            focal_id = unaccounted_veh_ids[0]
            unaccounted_veh_ids.remove(focal_id)
            
            # only want to center on existing agents
            origin_agent_idx = self.veh_id_to_idx[focal_id]
            if not self.states[origin_agent_idx, t, -1]:
                dead_agent_veh_ids.append(focal_id)
                continue

            road_points = preproc_data['road_points'].copy()
            road_types = preproc_data['road_types'].copy()

            if len(road_points) == 0:
                dead_agent_veh_ids.append(focal_id)
                continue

            data_veh_ids[focal_id] = [focal_id]
            
            rel_timesteps = np.repeat(np.expand_dims(timesteps, 0), self.cfg_rl_waymo.max_num_agents, axis=0)
            rel_timesteps = np.ones_like(rel_timesteps) * t  # so first timestep is 9, and so on.
            
            if t == self.history_steps - 1:
                self.relevant_agent_idxs[focal_id] = []

            if not ag_states[origin_agent_idx, normalize_timestep, -1]:
                dead_agent_veh_ids.append(focal_id)
                continue
            
            rel_ag_states, rel_ag_types, rel_actions, rel_rtgs, rel_goals, rel_moving_agent_mask, new_agent_idx_dict, relevant_agent_idxs = dset.select_relevant_agents(ag_states, ag_types, actions, rtgs, goals[:, 0], origin_agent_idx, normalize_timestep, moving_agent_mask, self.relevant_agent_idxs[focal_id])

            accounted_veh_ids = [self.idx_to_veh_id[idx] for idx in new_agent_idx_dict.keys()]
            for unacc_veh_id in unaccounted_veh_ids:
                if unacc_veh_id in accounted_veh_ids:
                    data_veh_ids[focal_id].append(unacc_veh_id)
                    unaccounted_veh_ids.remove(unacc_veh_id)

            if t == self.history_steps - 1:
                # all vehicles to evaluate within this set of 24 agents get the same context throughout trajectory
                # This is required due to possible edge case: focal_id exists at beginning (with context containing
                # another agent we wish to evaluate), but focal_id dies before this other agent dies.
                for veh_id in data_veh_ids[focal_id]:
                    self.relevant_agent_idxs[veh_id] = list(new_agent_idx_dict.keys())
            else:
                for veh_id in data_veh_ids[focal_id]:
                    self.relevant_agent_idxs[veh_id] = relevant_agent_idxs
            
            new_origin_agent_idx = new_agent_idx_dict[origin_agent_idx]
            rel_ag_states_future = rel_ag_states[:, 10:].copy()
            rel_ag_states = rel_ag_states[:, :10].copy()
            rel_road_points, rel_road_types = dset.select_indiv_agent_roads(rel_ag_states, road_points, road_types)

            yaw = rel_ag_states[:, -1, 4].copy()
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            translation = rel_ag_states[:, -1, :2].copy()
            translation_yaws = np.concatenate((translation, angle_of_rotation[:, None]), axis=-1)
            
            rel_ag_states, rel_ag_states_future, rel_road_points, rel_goals = dset._get_agents_local_frame(rel_ag_states, rel_ag_states_future, rel_road_points, rel_goals)
            rel_past_relative_encoding = dset._prepare_relative_encodings(rel_ag_states, rel_ag_states[:, -1:])
            if self.cfg_rl_waymo.future_relative_encoding:
                # use rel_ag_states_future here to compare
                rel_cv_future_states = dset._get_constant_velocity_futures(rel_ag_states[:, -1])
                rel_future_relative_encoding = dset._prepare_relative_encodings(rel_cv_future_states, rel_ag_states[:, -1:])
            else:
                rel_future_relative_encoding = rel_past_relative_encoding[:, :, -1:].repeat(self.cfg_rl_waymo.train_context_length-self.history_steps, axis=2)
            # remove global coordinates
            rel_ag_states = np.concatenate((rel_ag_states[:, :, 0:5], rel_ag_states[:, :, 10:]), axis=-1)
            if self.discretize_rtgs:
                rel_rtgs = dset.discretize_rtgs(rel_rtgs)

            rel_ag_states[:, :, :2] /= self.cfg_rl_waymo.state_normalizer.pos_div
            rel_ag_states[:, :, 2:4] /= self.cfg_rl_waymo.state_normalizer.vel_div
            rel_goals[:, :2] /= self.cfg_rl_waymo.state_normalizer.pos_div
            rel_goals[:, 2:4] /= self.cfg_rl_waymo.state_normalizer.vel_div
            rel_road_points[:, :, :, :2] /= self.cfg_rl_waymo.state_normalizer.pos_div
            rel_actions = dset._normalize_actions(rel_actions)

            # placeholder values
            rel_actions_future = rel_actions[:, -1:].repeat(self.cfg_rl_waymo.train_context_length-self.history_steps, axis=1)  
            rel_ag_states_future = rel_ag_states[:, -1:].repeat(self.cfg_rl_waymo.train_context_length-self.history_steps, axis=1)  
            
            d = dict()
            d['agent'] = from_numpy({
                'agent_past_states': add_batch_dim(rel_ag_states),
                'agent_past_actions': add_batch_dim(rel_actions),
                'agent_future_states': add_batch_dim(rel_ag_states_future),
                'agent_future_actions': add_batch_dim(rel_actions_future),
                'past_relative_encodings': add_batch_dim(rel_past_relative_encoding),
                'future_relative_encodings': add_batch_dim(rel_future_relative_encoding),
                'agent_types': add_batch_dim(rel_ag_types), 
                'goals': add_batch_dim(rel_goals),
                'rtgs': add_batch_dim(rel_rtgs),
                'timesteps': add_batch_dim(rel_timesteps),
                'moving_agent_mask': add_batch_dim(rel_moving_agent_mask),
                'agent_translation_yaws': add_batch_dim(translation_yaws)
            })
            d['map'] = from_numpy({
                'road_points': add_batch_dim(rel_road_points),
                'road_types': add_batch_dim(rel_road_types)
            })
            d = MotionData(d)

            motion_datas[focal_id] = d
            new_agent_idx_dicts[focal_id] = new_agent_idx_dict

        return motion_datas, dead_agent_veh_ids, new_agent_idx_dicts, data_veh_ids


    def predict(self, vehicle_data_dict, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t):
        motion_datas, dead_agent_veh_ids, new_agent_idx_dicts, data_veh_ids = self.get_data(gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t)
        
        # we don't want to predict an RTG that has already been predicted
        processed_rtg_veh_ids = []
        processed_rtgs_goal = {}
        processed_rtgs_veh = {}
        processed_rtgs_road = {}
        
        for focal_id in motion_datas.keys():
            data = motion_datas[focal_id].cuda()
            veh_ids_in_data = data_veh_ids[focal_id]

            focal_idx_in_model = new_agent_idx_dicts[focal_id][self.veh_id_to_idx[veh_ids_in_data[0]]]
            data['focal_idx_in_model'] = focal_idx_in_model

            # sample next action
            buffer_size = int(10 / self.sampling_frequency)
            pred_actions = self.model(data)[0, :, :buffer_size, -2:]
            pred_actions = dset._unnormalize_actions(pred_actions)
            for veh_id in veh_ids_in_data:
                pred_actions_idx = new_agent_idx_dicts[focal_id][self.veh_id_to_idx[veh_id]]
                next_pred_action = pred_actions[pred_actions_idx, :buffer_size].cpu().numpy()
                for b in range(buffer_size):
                    if b == 0:
                        vehicle_data_dict[veh_id][self.key_dict['next_acceleration']] = [next_pred_action[b, 0]]
                        vehicle_data_dict[veh_id][self.key_dict['next_steering']] = [next_pred_action[b, 1]]
                    else:
                        vehicle_data_dict[veh_id][self.key_dict['next_acceleration']].append(next_pred_action[b, 0])
                        vehicle_data_dict[veh_id][self.key_dict['next_steering']].append(next_pred_action[b, 1])


        for veh_id in dead_agent_veh_ids:
            buffer_size = int(10 / self.sampling_frequency)
            for b in range(buffer_size):
                if b == 0:
                    vehicle_data_dict[veh_id][self.key_dict['next_acceleration']] = [0.]
                    vehicle_data_dict[veh_id][self.key_dict['next_steering']] = [0.]
                else:
                    vehicle_data_dict[veh_id][self.key_dict['next_acceleration']].append(0.)
                    vehicle_data_dict[veh_id][self.key_dict['next_steering']].append(0.)

        return vehicle_data_dict


    def act(self, veh, t, vehicle_data_dict):
        veh_id = veh.getID()
        veh_exists = vehicle_data_dict[veh_id]['existence'][-1]
        
        buffer_idx = (t + 1) % int(10 / self.sampling_frequency)
        if not veh_exists:
            acceleration = 0.0
            steering = 0.0
            veh.setPosition(-1000000, -1000000)  # make cars disappear if they are out of actions
        else:
            acceleration = vehicle_data_dict[veh_id][self.key_dict['next_acceleration']][buffer_idx]
            steering = vehicle_data_dict[veh_id][self.key_dict['next_steering']][buffer_idx]

        if acceleration > 0.0:
            veh.acceleration = acceleration
        else:
            veh.brake(np.abs(acceleration))
        veh.steering = steering

        return veh, [acceleration, steering]