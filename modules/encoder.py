import torch
import torch.nn as nn

from utils.train_utils import weight_init
from utils.layers import MLPLayer
import torch.nn.functional as F 
from modules.map_encoder import MapEncoder

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_rl_waymo = self.cfg.dataset.waymo
        self.action_dim = self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization

        if self.cfg_model.use_map:
            self.map_encoder = MapEncoder(self.cfg)

        self.embed_state = MLPLayer(self.cfg_model.state_dim, self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)
        self.embed_goal = MLPLayer(self.cfg_rl_waymo.goal_dim, self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)
        self.embed_state_goal = nn.Linear(self.cfg_model.hidden_dim * 2, self.cfg_model.hidden_dim)
        self.embed_action = nn.Embedding(int(self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization), 
                                            self.cfg_model.hidden_dim)
        
        if self.cfg_model.decision_transformer:
            self.embed_rtg_goal = nn.Linear(1, self.cfg_model.hidden_dim)
            self.embed_rtg_veh = nn.Linear(1, self.cfg_model.hidden_dim)
            self.embed_rtg_road = nn.Linear(1, self.cfg_model.hidden_dim)
        else:
            self.embed_rtg_goal = nn.Embedding(self.cfg_rl_waymo.rtg_discretization, self.cfg_model.hidden_dim)
            self.embed_rtg_veh = nn.Embedding(self.cfg_rl_waymo.rtg_discretization, self.cfg_model.hidden_dim)
            self.embed_rtg_road = nn.Embedding(self.cfg_rl_waymo.rtg_discretization, self.cfg_model.hidden_dim)
        
        self.embed_rtg = nn.Linear(self.cfg_model.hidden_dim * self.cfg_model.num_reward_components, self.cfg_model.hidden_dim)

        self.embed_timestep = nn.Embedding(self.cfg_rl_waymo.max_timestep, self.cfg_model.hidden_dim)
        self.embed_agent_id = nn.Embedding(self.cfg_rl_waymo.max_num_agents, self.cfg_model.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.cfg_model.hidden_dim)

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.cfg_model.hidden_dim, 
                                                                                    nhead=self.cfg_model.num_heads,
                                                                                    dim_feedforward=self.cfg_model.dim_feedforward,
                                                                                    batch_first=True), 
                                                                                    num_layers=self.cfg_model.num_transformer_encoder_layers)
        self.apply(weight_init)


    def forward(self, data, eval):
        # focal_idx_in_model = data['focal_idx_in_model']
        agent_states = data['agent'].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]
        existence_mask = agent_states[:, :, :, -1:]
        agent_types = data['agent'].agent_types
        goals = data['agent'].goals
            
        actions = data['agent'].actions
        agent_ids = torch.arange(self.cfg_rl_waymo.max_num_agents).to(agent_states.device)
        # [batch_size, n_agents, timesteps]
        agent_ids = agent_ids.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, agent_states.shape[2])
        rtgs = data['agent'].rtgs
        timesteps = data['agent'].timesteps

        # [batch_size, timesteps, n_agents, 5]
        agent_types = agent_types.unsqueeze(2).repeat(1, 1, agent_states.shape[2], 1).transpose(1,2)
        # [batch_size, timesteps, n_agents, 5]
        goals = goals.unsqueeze(2).repeat(1, 1, agent_states.shape[2], 1).transpose(1,2)

        # [batch_size, timesteps, n_agents, num_actions]
        actions = actions.transpose(1,2)
        # [batch_size, timesteps, n_agents, num_reward_components]
        rtgs = rtgs.transpose(1,2)
        # [batch_size, timesteps, n_agents, 7]
        agent_states = agent_states[:, :, :, :-1].transpose(1,2)
        # [batch_size, timesteps, n_agents, 1]
        timesteps = timesteps.transpose(1,2)
        agent_ids = agent_ids.transpose(1,2)
        # [batch_size, timesteps, n_agents, 1]
        existence_mask = existence_mask.transpose(1,2)
        states = torch.cat([agent_states, agent_types], dim=-1)
        
        if self.cfg_model.encode_initial_state:
            initial_states = states[:, 0].float()
            initial_existence_mask = existence_mask[:, 0]

        existence_mask = existence_mask.reshape(batch_size,seq_len*self.cfg_rl_waymo.max_num_agents, 1)
        timesteps =  timesteps.reshape(batch_size,seq_len*self.cfg_rl_waymo.max_num_agents)
        agent_ids = agent_ids.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents)
        states = states.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, self.cfg_model.state_dim).float() 
        goals = goals[:, :, :, :self.cfg_rl_waymo.goal_dim].reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, self.cfg_rl_waymo.goal_dim).float()
        actions = actions.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents)
        if self.cfg_model.decision_transformer:
            rtgs = rtgs.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, self.cfg_model.num_reward_components).float()
        else:
            rtgs = rtgs.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, self.cfg_model.num_reward_components)

        timestep_embeddings = self.embed_timestep(timesteps)
        agent_id_embeddings = self.embed_agent_id(agent_ids)
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        
        if not eval:
            goal_dropout_mask = torch.rand(size=(batch_size, self.cfg_rl_waymo.max_num_agents), device=goal_embeddings.device) > self.cfg_model.goal_dropout
            goal_dropout_mask = goal_dropout_mask.unsqueeze(1).repeat(1, seq_len, 1).reshape(batch_size, -1, 1).float()
            goal_embeddings = goal_embeddings * goal_dropout_mask
        
        state_embeddings = self.embed_state_goal(torch.cat([state_embeddings, goal_embeddings], dim=-1)) + timestep_embeddings + agent_id_embeddings
        
        if self.cfg_model.encode_initial_state:
            initial_state_embeddings = state_embeddings[:, 0:self.cfg_rl_waymo.max_num_agents]
        
        action_embeddings = self.embed_action(actions.long()) + timestep_embeddings + agent_id_embeddings
        
        if self.cfg_model.decision_transformer:
            rtg_goal_embeddings = self.embed_rtg_goal(rtgs[:, :, 0:1])
            rtg_veh_embeddings = self.embed_rtg_veh(rtgs[:, :, 1:2])
            rtg_road_embeddings = self.embed_rtg_road(rtgs[:, :, 2:3])
            rtg_embeddings = self.embed_rtg(torch.cat([rtg_goal_embeddings, rtg_veh_embeddings, rtg_road_embeddings], dim=-1)) + timestep_embeddings + agent_id_embeddings
        else:
            rtg_goal_embeddings = self.embed_rtg_goal(rtgs[:, :, 0].long())
            rtg_veh_embeddings = self.embed_rtg_veh(rtgs[:, :, 1].long())
            rtg_road_embeddings = self.embed_rtg_road(rtgs[:, :, 2].long())
            rtg_embeddings = self.embed_rtg(torch.cat([rtg_goal_embeddings, rtg_veh_embeddings, rtg_road_embeddings], dim=-1)) + timestep_embeddings + agent_id_embeddings

        # zero out embeddings for missing timesteps
        state_embeddings = state_embeddings * existence_mask.float()
        if self.cfg_model.no_actions:
            action_embeddings = action_embeddings * torch.zeros_like(existence_mask.float())
        else:
            action_embeddings = action_embeddings * existence_mask.float()
        rtg_embeddings = rtg_embeddings * existence_mask.float()
        
        if self.cfg_model.encode_initial_state:
            initial_state_embeddings = initial_state_embeddings * initial_existence_mask.float()
            initial_existence_mask = initial_existence_mask[:, :, 0].bool()

        if self.cfg_model.decision_transformer:
            stacked_embeddings = torch.stack(
                (rtg_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents*3, self.cfg_model.hidden_dim)
        elif self.cfg_model.trajeglish:
            stacked_embeddings = action_embeddings.unsqueeze(1).permute(0, 2, 1, 3).reshape(batch_size, 1*seq_len*self.cfg_rl_waymo.max_num_agents, self.cfg_model.hidden_dim)
        elif self.cfg_model.il:
            stacked_embeddings = torch.stack(
                (state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents*2, self.cfg_model.hidden_dim)
        else:
            stacked_embeddings = torch.stack(
                (state_embeddings, rtg_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents*3, self.cfg_model.hidden_dim)
        stacked_embeddings = self.embed_ln(stacked_embeddings)

        if self.cfg_model.use_map:
            polyline_embeddings, valid_mask = self.map_encoder(data)
            valid_mask = valid_mask.bool()
            
            if self.cfg_model.encode_initial_state:
                pre_encoder_embeddings = torch.cat([polyline_embeddings, initial_state_embeddings], dim=1)
                # we use ~ as "True" corresponds to "ignored" in src_key_padding_mask
                src_key_padding_mask = ~torch.cat([valid_mask, initial_existence_mask], dim=1)
            else:
                pre_encoder_embeddings = polyline_embeddings
                # we use ~ as "True" corresponds to "ignored" in src_key_padding_mask
                src_key_padding_mask = ~valid_mask

            encoder_embeddings = self.transformer_encoder(pre_encoder_embeddings, src_key_padding_mask=src_key_padding_mask)
        
        else:
            src_key_padding_mask = ~initial_existence_mask
            encoder_embeddings = self.transformer_encoder(initial_state_embeddings, src_key_padding_mask=src_key_padding_mask)
        
        return {
            'stacked_embeddings': stacked_embeddings, 
            'encoder_embeddings': encoder_embeddings, 
            'src_key_padding_mask': src_key_padding_mask
        }