import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from torch_geometric.data import Data 
from torch_geometric.data import Batch 

from typing import List, Optional
import math
import numpy as np
import torch.nn.functional as F
from typing import List, Optional
from utils.layers import MLPLayer
from utils.train_utils import weight_init
from utils.diffusion_helpers import SinusoidalPosEmb
from modules.ctg_arch import MapEncoderPtsMA, RelativeSocialAttentionLayer, SingleInputEmbedding, PositionalEncoding


class RTGModel(nn.Module):
    def __init__(self, cfg):
        super(RTGModel, self).__init__()

        self.cfg_rl_waymo = cfg.dataset.waymo
        self.cfg_model = cfg.model

        self.k_attr = self.cfg_rl_waymo.k_attr
        self.tgt_k_attr = self.k_attr - 2  # removing width and length
        self.map_attr = self.cfg_rl_waymo.map_attr
        self.action_attr = self.cfg_rl_waymo.action_dim
        self.goal_dim = self.cfg_rl_waymo.goal_dim
        self.reward_attr = self.cfg_rl_waymo.num_reward_components
        self.num_road_types = self.cfg_rl_waymo.num_road_types
        self.num_agents = self.cfg_rl_waymo.max_num_agents
        self.num_agent_types = self.cfg_rl_waymo.num_agent_types

        self.diffusion_type = self.cfg_model.diffusion_type
        self.d_k = self.cfg_model.hidden_dim
        self.L_enc = self.cfg_model.num_transformer_encoder_layers
        self.dropout = self.cfg_model.dropout
        self.num_heads = self.cfg_model.num_heads
        self.tx_hidden_size = self.cfg_model.dim_feedforward
        self.k_attr_rel = 7
        self.num_freq_bands = 64
        self.goal_dropout = self.cfg_model.goal_dropout

        self.embed_state_action = MLPLayer(self.k_attr+self.num_agent_types+self.action_attr, self.d_k, self.d_k)
        self.embed_goal = MLPLayer(self.goal_dim, self.d_k, self.d_k)
        self.embed_timestep = nn.Embedding(self.cfg_rl_waymo.max_timestep, self.d_k)

        self.predict_rtg = MLPLayer(self.d_k, self.d_k, self.cfg_rl_waymo.rtg_discretization * self.cfg_model.num_reward_components)

        self.embed_all_elements = MLPLayer(self.d_k * 2, self.d_k, self.d_k)  # concat state, goal, rtgs 

        self.relative_encodings_encoder = SingleInputEmbedding(self.k_attr_rel, self.d_k)

        self.sincos_emb = SinusoidalPosEmb(self.d_k)
        self.diffustion_step_encoder = MLPLayer(self.d_k, self.d_k, self.d_k)
        
        self.map_encoder = MapEncoderPtsMA(cfg=cfg)

        self.social_attn_layers = []
        self.temporal_attn_layers = []
        self.map_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size,
                                                          batch_first=False)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            self.social_attn_layers.append(RelativeSocialAttentionLayer(d_model=self.d_k, nhead=self.num_heads, 
                                                                        dropout=self.dropout, dim_feedforward=self.tx_hidden_size))
            
            self.map_attn_layers.append(nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)
        self.map_attn_layers = nn.ModuleList(self.map_attn_layers)

        self.pos_encoder = PositionalEncoding(d_model=self.d_k, dropout=self.dropout, max_len=100)
        
        self.apply(weight_init)

        self.train()

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * self.num_agents, -1)), 
                                src_key_padding_mask=agent_masks)
        return agents_temp_emb.view(T_obs, B, self.num_agents, -1)
    
    def social_attn_fn(self, agents_emb, agent_masks, relative_encodings_emb, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :param relative_encodings_emb: (B, T, N, N, H)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self.num_agents, B * T_obs, -1)
        relative_encodings_emb = relative_encodings_emb.reshape(B * T_obs, (self.num_agents) * (self.num_agents), -1)
        agents_soc_emb = layer(agents_emb, relative_encodings_emb, src_key_padding_mask=agent_masks.reshape(-1, self.num_agents))
        agents_soc_emb = agents_soc_emb.reshape(self.num_agents, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def map_attn_fn(self, agents_emb, map_features, road_masks, layer):
        T_obs, B, _, _ = agents_emb.shape
        S = map_features.shape[0]
        agents_emb = agents_emb.reshape(T_obs, -1, self.d_k)
        map_features = map_features.reshape(S, -1, self.d_k)
        road_masks = road_masks.reshape(-1, S)
        agents_emb_1 = layer(query=agents_emb, key=map_features, value=map_features,
                             key_padding_mask=road_masks)[0]
        agents_emb = agents_emb + agents_emb_1
        agents_emb = agents_emb.reshape(T_obs, B, self.num_agents, -1)
        return agents_emb

    def forward(self, cond, eval=False):
        (agent_past_states, agent_past_actions, agent_past_rel_encodings, agent_future_rel_encodings, 
                agent_types, goals, timesteps, rtgs, road_points, road_types, moving_agent_masks) = cond
        
        batch_size, num_agents, in_horizon, _ = agent_past_states.shape

        agent_types = agent_types.unsqueeze(2).repeat(1, 1, in_horizon, 1)  # [B, N, Tin+Tout, 5]
        past_state_actions = torch.cat((agent_past_states[..., :-1], agent_past_actions), dim=-1)  # [B, N, Tin, 9]
        agent_past_existence = agent_past_states[..., -1].permute(0, 2, 1)
        agent_existence = (1.0 - agent_past_existence).type(torch.BoolTensor).to(agent_past_states.device)

        past_future_tensor = torch.cat((past_state_actions, agent_types), dim=-1)

        # Embedding stuff
        state_act_emb = self.embed_state_action(past_future_tensor)
        goal_emb = self.embed_goal(goals).unsqueeze(2).repeat(1, 1, in_horizon, 1)
        if not eval:
            goal_dropout_mask = torch.rand(size=(batch_size, self.cfg_rl_waymo.max_num_agents, 1, 1), device=goal_emb.device) > self.cfg_model.goal_dropout
            goal_emb = goal_emb * goal_dropout_mask

        timestep_emb = self.embed_timestep(timesteps)[:, :, :in_horizon, 0]
        agent_embedding = self.embed_all_elements(torch.cat((state_act_emb, goal_emb), dim=-1)) + timestep_emb
        agent_embedding = agent_embedding.permute(2, 0, 1, 3)

        #[B, T_obs, M, M, k_attr_rel]
        relative_encodings_emb = self.relative_encodings_encoder(agent_past_rel_encodings).permute(0, 3, 1, 2, 4)
        
        # Process through AutoBot's encoder
        map_features, road_segs_masks = self.map_encoder(road_points, road_types)
        for i in range(self.L_enc):
            agent_embedding = self.temporal_attn_fn(agent_embedding, agent_existence, layer=self.temporal_attn_layers[i])
            agent_embedding = self.social_attn_fn(agent_embedding, agent_existence, relative_encodings_emb, layer=self.social_attn_layers[i])
            agent_embedding = self.map_attn_fn(agent_embedding, map_features, road_segs_masks, layer=self.map_attn_layers[i])

        pred_rtgs = self.predict_rtg(agent_embedding[-1])
        return pred_rtgs

    def loss(self, cond):
        pred_rtgs = self(cond)
        (agent_past_states, agent_past_actions, agent_past_rel_encodings, agent_future_rel_encodings, 
            agent_types, goals, timesteps, rtgs, road_points, road_types, moving_agent_masks) = cond
        existence_mask = agent_past_states[..., -1, -1].reshape(-1)

        rtg_preds = pred_rtgs.reshape(-1, self.cfg_rl_waymo.rtg_discretization, self.cfg_model.num_reward_components)
        rtg_goal_logits = rtg_preds[:, :, 0]
        rtg_veh_logits = rtg_preds[:, :, 1]
        rtg_goal = rtgs[:, :, -1, 0].reshape(-1)
        rtg_veh = rtgs[:, :, -1, 1].reshape(-1)
        
        loss_rtg_goal = F.cross_entropy(rtg_goal_logits.float(), rtg_goal.long(), reduction='none')
        loss_rtg_goal = loss_rtg_goal * existence_mask.float()
        loss_rtg_goal = loss_rtg_goal.sum() / existence_mask.sum()

        loss_rtg_veh = F.cross_entropy(rtg_veh_logits.float(), rtg_veh.long(), reduction='none')
        loss_rtg_veh = loss_rtg_veh * existence_mask.float()
        loss_rtg_veh = loss_rtg_veh.sum() / existence_mask.sum()

        rtg_road_logits = rtg_preds[:, :, 2]
        rtg_road = rtgs[:, :, -1, 2].reshape(-1)
        loss_rtg_road = F.cross_entropy(rtg_road_logits.float(), rtg_road.long(), reduction='none')
        loss_rtg_road = loss_rtg_road * existence_mask.float()
        loss_rtg_road = loss_rtg_road.sum() / existence_mask.sum()

        return loss_rtg_goal, loss_rtg_veh, loss_rtg_road