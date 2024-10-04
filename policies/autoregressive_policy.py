from utils.data import get_object_type_onehot, add_batch_dim, from_numpy, MotionData
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import time
from policies.policy import Policy

class AutoregressivePolicy(Policy):
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
                 action_temperature, 
                 nucleus_sampling, 
                 nucleus_threshold):
        
        super(AutoregressivePolicy, self).__init__(cfg, 
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
        
        self.action_temperature = action_temperature 
        self.nucleus_sampling = nucleus_sampling 
        self.nucleus_threshold = nucleus_threshold 
        if tilt_dict['tilt']:
            self.goal_tilt = tilt_dict['goal_tilt']
            self.veh_veh_tilt = tilt_dict['veh_veh_tilt']
            self.veh_edge_tilt = tilt_dict['veh_edge_tilt']


    def get_data(self, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t):
        moving_ids = np.where(np.linalg.norm(self.states[:, 0, :2] - self.goals[:, 0, :2], axis=1) > self.cfg_rl_waymo.moving_threshold)[0]
        moving_agent_mask = np.isin(np.arange(self.states.shape[0]), moving_ids)
        
        if t < self.cfg_rl_waymo.train_context_length:
            # retrieve most recent states in the context
            ag_states = self.states[:, :self.cfg_rl_waymo.train_context_length].copy()
            ag_types = self.types.copy()
            actions = self.actions[:, :self.cfg_rl_waymo.train_context_length].copy()
            rtgs = self.rtgs[:, :self.cfg_rl_waymo.train_context_length].copy()
            goals = self.goals[:, :self.cfg_rl_waymo.train_context_length].copy()
            timesteps = self.timesteps[0, :self.cfg_rl_waymo.train_context_length].astype(int).copy()
        else:
            # retrieve most recent states in the context
            ag_states = self.states[:,t-(self.cfg_rl_waymo.train_context_length - 1):t+1].copy()
            ag_types = self.types.copy()
            actions = self.actions[:, t-(self.cfg_rl_waymo.train_context_length - 1):t+1].copy()
            rtgs = self.rtgs[:, t-(self.cfg_rl_waymo.train_context_length - 1):t+1].copy()
            goals = self.goals[:, t-(self.cfg_rl_waymo.train_context_length - 1):t+1].copy()
            timesteps = self.timesteps[0, t-(self.cfg_rl_waymo.train_context_length - 1):t+1].astype(int).copy()

        normalize_timestep = 0
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
            
            if t == 0:
                self.relevant_agent_idxs[focal_id] = []
            
            rel_ag_states, rel_ag_types, rel_actions, rel_rtgs, rel_goals, rel_moving_agent_mask, new_agent_idx_dict, relevant_agent_idxs = dset.select_relevant_agents(ag_states, ag_types, actions, rtgs, goals[:, 0], origin_agent_idx, normalize_timestep, moving_agent_mask, self.relevant_agent_idxs[focal_id])

            
            accounted_veh_ids = [self.idx_to_veh_id[idx] for idx in new_agent_idx_dict.keys()]
            for unacc_veh_id in unaccounted_veh_ids:
                if unacc_veh_id in accounted_veh_ids:
                    data_veh_ids[focal_id].append(unacc_veh_id)
                    unaccounted_veh_ids.remove(unacc_veh_id)

            if t == 0:
                # all vehicles to evaluate within this set of 24 agents get the same context throughout trajectory
                # This is required due to possible edge case: focal_id exists at beginning (with context containing
                # another agent we wish to evaluate), but focal_id dies before this other agent dies.
                for veh_id in data_veh_ids[focal_id]:
                    self.relevant_agent_idxs[veh_id] = list(new_agent_idx_dict.keys())
            else:
                for veh_id in data_veh_ids[focal_id]:
                    self.relevant_agent_idxs[veh_id] = relevant_agent_idxs
            
            new_origin_agent_idx = new_agent_idx_dict[origin_agent_idx]
            rel_actions = dset.discretize_actions(rel_actions)
            if self.discretize_rtgs:
                rel_rtgs = dset.discretize_rtgs(rel_rtgs)
            rel_ag_states, rel_road_points, rel_road_types, rel_goals = dset.normalize_scene(rel_ag_states, road_points, road_types, rel_goals, new_origin_agent_idx)       
            
            d = dict()
            # need to add batch dim as pytorch_geometric batches along first dimension of torch Tensors
            d['agent'] = from_numpy({
                'agent_states': add_batch_dim(rel_ag_states),
                'agent_types': add_batch_dim(rel_ag_types), 
                'goals': add_batch_dim(rel_goals),
                'actions': add_batch_dim(rel_actions),
                'rtgs': add_batch_dim(rel_rtgs),
                'timesteps': add_batch_dim(rel_timesteps), # TODO: clean this up
                'moving_agent_mask': add_batch_dim(rel_moving_agent_mask)
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
        
        if t < self.cfg_rl_waymo.train_context_length:
            token_index = t 
        else:
            token_index = -1
        
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
            
            if self.predict_rtgs:
                preds = self.model(data, eval=True)
                rtg_logits = preds['rtg_preds']
                
                veh_ids_in_context = [self.idx_to_veh_id[idx] for idx in self.relevant_agent_idxs[focal_id]]
                # append predicted RTG to data dictionary before making action prediction
                for veh_id in veh_ids_in_context:
                    if veh_id in processed_rtg_veh_ids:
                        data['agent'].rtgs[0, new_agent_idx_dicts[focal_id][self.veh_id_to_idx[veh_id]], token_index, 0] = processed_rtgs_goal[veh_id]
                        data['agent'].rtgs[0, new_agent_idx_dicts[focal_id][self.veh_id_to_idx[veh_id]], token_index, 1] = processed_rtgs_veh[veh_id]
                        data['agent'].rtgs[0, new_agent_idx_dicts[focal_id][self.veh_id_to_idx[veh_id]], token_index, 2] = processed_rtgs_road[veh_id] 
                        continue
                    
                    vehicle_data_dict, data, next_rtgs = self.process_predicted_rtg(rtg_logits, token_index, veh_id, dset, vehicle_data_dict, data, new_agent_idx_dicts[focal_id], is_tilted=veh_id in veh_ids_in_data)
                    processed_rtg_veh_ids.append(veh_id)
                    assert veh_id not in processed_rtgs_goal.keys()
                    processed_rtgs_goal[veh_id] = next_rtgs[0]
                    processed_rtgs_veh[veh_id] = next_rtgs[1]
                    processed_rtgs_road[veh_id] = next_rtgs[2]

            # sample next action
            preds = self.model(data, eval=True)
            # [batch_size=1, num_agents, timesteps, action_dim]
            logits = preds['action_preds']
            
            for veh_id in veh_ids_in_data:
                next_action_logits = logits[0, new_agent_idx_dicts[focal_id][self.veh_id_to_idx[veh_id]], token_index]
                
                if self.nucleus_sampling:
                    action_probs = F.softmax(next_action_logits / self.action_temperature, dim=0)
                    sorted_probs, sorted_indices = torch.sort(action_probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    selected_actions = cum_probs < self.nucleus_threshold
                    # To include the next token so that we minimize the cumsum >= p
                    selected_actions = torch.cat([selected_actions.new_ones(selected_actions.shape[:-1] + (1,)), selected_actions[..., :-1]], dim=-1)
                    
                    # Keep top-p probs
                    new_probs = sorted_probs[selected_actions]
                    # Re-normalize the probs
                    new_probs /= new_probs.sum()

                    next_action_dis = torch.zeros_like(next_action_logits)
                    next_action_dis[sorted_indices[selected_actions]] = new_probs
                else:
                    next_action_dis = F.softmax(next_action_logits / self.action_temperature, dim=0)
                
                # sample from output distribution
                next_action = torch.multinomial(next_action_dis, 1)
                next_action = next_action.reshape(1, 1)
                next_action_continuous = dset.undiscretize_actions(next_action.cpu().numpy())
                vehicle_data_dict[veh_id][self.key_dict['next_acceleration']] = next_action_continuous[0,0,0]
                vehicle_data_dict[veh_id][self.key_dict['next_steering']] = next_action_continuous[0,0,1]

        if self.predict_rtgs:
            for veh_id in vehicle_data_dict.keys():
                if veh_id in processed_rtg_veh_ids:
                    vehicle_data_dict[veh_id][self.key_dict['rtgs']].append(np.array([vehicle_data_dict[veh_id]['next_rtg_goal'], vehicle_data_dict[veh_id]['next_rtg_veh'], vehicle_data_dict[veh_id]['next_rtg_road']]))
                else:
                    vehicle_data_dict[veh_id][self.key_dict['rtgs']].append(np.array([0] * self.cfg_model.num_reward_components))
        
        for veh_id in dead_agent_veh_ids:
            vehicle_data_dict[veh_id][self.key_dict['next_acceleration']] = 0.
            vehicle_data_dict[veh_id][self.key_dict['next_steering']] = 0.

        return vehicle_data_dict


    def act(self, veh, t, vehicle_data_dict):
        veh_id = veh.getID()
        veh_exists = vehicle_data_dict[veh_id]['existence'][-1]
        
        if not veh_exists:
            acceleration = 0.0
            steering = 0.0
            veh.setPosition(-1000000, -1000000)  # make cars disappear if they are out of actions
        else:
            acceleration = vehicle_data_dict[veh_id][self.key_dict['next_acceleration']]
            steering = vehicle_data_dict[veh_id][self.key_dict['next_steering']]

        if acceleration > 0.0:
            veh.acceleration = acceleration
        else:
            veh.brake(np.abs(acceleration))
        veh.steering = steering

        return veh, [acceleration, steering]