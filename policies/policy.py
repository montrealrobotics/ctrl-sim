from utils.data import get_object_type_onehot, add_batch_dim, from_numpy, MotionData
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import time

class Policy:
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
                 name):
        
        self.cfg = cfg.copy()
        self.model_path = model_path
        self.model = model
        self.model.eval()
        self.cfg_model = model.cfg.model
        self.cfg_rl_waymo = model.cfg.dataset.waymo
        
        self.steps = self.cfg.nocturne.steps
        self.model.eval()
        self.use_rtg = use_rtg 
        self.predict_rtgs = predict_rtgs
        self.discretize_rtgs = discretize_rtgs
        self.real_time_rewards = real_time_rewards
        self.privileged_return = privileged_return 
        self.max_return = max_return 
        self.min_return = min_return
        self.key_dict = key_dict 
        self.tilt_dict = tilt_dict 
        self.name = name

    
    def reset(self, vehicle_data_dict):
        num_agents = len(vehicle_data_dict.keys())
        self.states = np.zeros((num_agents, self.steps, 8))
        self.gt_states = np.zeros((num_agents, self.steps, 8)) # unused in autoregressive_policy
        self.types = np.zeros((num_agents, 5))
        self.actions = np.zeros((num_agents, self.steps, 2)) # acceleration and steering
        self.rtgs = np.zeros((num_agents, self.steps, self.cfg_model.num_reward_components))
        self.goals = np.zeros((num_agents, self.steps, self.cfg_rl_waymo.goal_dim))
        self.timesteps = np.zeros((num_agents, self.steps, 1))
        self.relevant_agent_idxs = {}
        self.idx_to_veh_id = {}
        self.veh_id_to_idx = {}
        for i,v in enumerate(vehicle_data_dict.keys()):
            self.idx_to_veh_id[i] = v
            self.veh_id_to_idx[v] = i

    
    def update_gt_state(self, gt_data_dict):
        for i, v in enumerate(gt_data_dict.keys()):
            traj = np.array(gt_data_dict[v]['traj'])
            self.gt_states[i] = traj[:self.steps]


    def update_state(self, vehicle_data_dict, vehicles_to_evaluate, t):
        for i, v in enumerate(vehicle_data_dict.keys()):
            position_x = np.array([vehicle_data_dict[v]["position"][t]['x']])
            position_y = np.array([vehicle_data_dict[v]["position"][t]['y']])
            yaw = np.array([vehicle_data_dict[v]["heading"][t]])
            velocity_x = np.array([vehicle_data_dict[v]["velocity"][t]['x']])
            velocity_y = np.array([vehicle_data_dict[v]["velocity"][t]['y']])
            length = np.array([vehicle_data_dict[v]['length']])
            width = np.array([vehicle_data_dict[v]['width']])
            existence = np.array([vehicle_data_dict[v]["existence"][t]])
            state = np.concatenate([position_x, position_y, velocity_x, velocity_y, yaw, length, width, existence], axis=0)
            self.states[i, t] = state

            if t == 0:
                self.types[i] = get_object_type_onehot(vehicle_data_dict[v]["type"])
            self.timesteps[i, t] = np.array([vehicle_data_dict[v]["timestep"][t]])

            if t > 0:
                acceleration = np.array([vehicle_data_dict[v]["acceleration"][t-1]])
                steering = np.array([vehicle_data_dict[v]["steering"][t-1]])
                action = np.concatenate([acceleration, steering], axis=0)
                self.actions[i, t-1] = action
                if self.use_rtg:
                    rtg = np.array([vehicle_data_dict[v][self.key_dict["rtgs"]][t-1]])[0]
                    self.rtgs[i, t-1] = rtg

            if self.real_time_rewards and self.use_rtg:
                rtg = np.array([vehicle_data_dict[v][self.key_dict["rtgs"]][t]])[0]
                self.rtgs[i, t] = rtg

            goal_position_x = np.array([vehicle_data_dict[v]["goal_position"]['x']])
            goal_position_y = np.array([vehicle_data_dict[v]["goal_position"]['y']])
            goal_heading = np.array([vehicle_data_dict[v]["goal_heading"]])
            goal_speed = np.array([vehicle_data_dict[v]["goal_speed"]])
            goal_velocity_x = goal_speed * np.cos(goal_heading)
            goal_velocity_y = goal_speed * np.sin(goal_heading)
            goal = np.concatenate([goal_position_x, goal_position_y, goal_velocity_x, goal_velocity_y, goal_heading], axis=0)
            self.goals[i, t] = goal[:self.cfg_rl_waymo.goal_dim]

    
    def process_predicted_rtg(self, rtg_logits, token_index, veh_id, dset, vehicle_data_dict, data, new_agent_idx_dict, is_tilted=False):
        idx = new_agent_idx_dict[self.veh_id_to_idx[veh_id]]
        
        next_rtg_logits = rtg_logits[0, idx, token_index].reshape(self.cfg_rl_waymo.rtg_discretization, self.cfg_model.num_reward_components)
        next_rtg_goal_logits = next_rtg_logits[:, 0]
        next_rtg_veh_logits = next_rtg_logits[:, 1]
        next_rtg_road_logits = next_rtg_logits[:, 2]
        
        # is_tilted is whether we tilt the specific agent and self.tilt_dict['tilt'] is whether the model supports tilting
        if is_tilted and self.tilt_dict['tilt']:
            tilt_logits = torch.from_numpy(dset.get_tilt_logits(self.tilt_dict['goal_tilt'], self.tilt_dict['veh_veh_tilt'], self.tilt_dict['veh_edge_tilt'])).cuda()
        else:
            tilt_logits = torch.from_numpy(dset.get_tilt_logits(0, 0, 0)).cuda()

        next_rtg_goal_dis = F.softmax(next_rtg_goal_logits + tilt_logits[:, 0], dim=0)
        next_rtg_goal = torch.multinomial(next_rtg_goal_dis, 1)
        next_rtg_veh_dis = F.softmax(next_rtg_veh_logits + tilt_logits[:, 1], dim=0)
        next_rtg_veh = torch.multinomial(next_rtg_veh_dis, 1)
        next_rtg_road_dis = F.softmax(next_rtg_road_logits + tilt_logits[:, 2], dim=0)
        next_rtg_road = torch.multinomial(next_rtg_road_dis, 1)
        
        next_rtg = torch.cat([next_rtg_goal.reshape(1, 1, 1), next_rtg_veh.reshape(1, 1, 1), next_rtg_road.reshape(1, 1, 1)], dim=2)
        next_rtg_continuous = dset.undiscretize_rtgs(next_rtg.cpu().numpy())
        vehicle_data_dict[veh_id]['next_rtg_goal'] = next_rtg_continuous[0,0,0]
        vehicle_data_dict[veh_id]['next_rtg_veh'] = next_rtg_continuous[0,0,1]
        vehicle_data_dict[veh_id]['next_rtg_road'] = next_rtg_continuous[0,0,2]

        # append predicted RTG to data dictionary before making action prediction
        data['agent'].rtgs[0, idx, token_index, 0] = next_rtg_goal
        data['agent'].rtgs[0, idx, token_index, 1] = next_rtg_veh
        data['agent'].rtgs[0, idx, token_index, 2] = next_rtg_road

        next_rtgs = [next_rtg_goal, next_rtg_veh, next_rtg_road]

        return vehicle_data_dict, data, next_rtgs
    
    
    def get_data(self, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t):
        pass

    
    def predict(self, vehicle_data_dict, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t):
        pass


    def act(self, veh, t, vehicle_data_dict):
        pass


    
        
