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


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RelativeSocialAttentionLayer(MessagePassing):

    def __init__(self,
                 d_model,
                 nhead,
                 dropout,
                 dim_feedforward):

        super(RelativeSocialAttentionLayer, self).__init__()

        self.d_model = d_model 
        self.nhead = nhead 
        self.dim_feedforward = dim_feedforward 

        self.lin_q_node = nn.Linear(d_model, d_model)
        self.lin_k_node = nn.Linear(d_model, d_model)
        self.lin_k_edge = nn.Linear(d_model, d_model)
        self.lin_v_node = nn.Linear(d_model, d_model)
        self.lin_v_edge = nn.Linear(d_model, d_model)

        self.lin_self = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.lin_ih = nn.Linear(d_model, d_model)
        self.lin_hh = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout))

    def forward(self,
                x,
                edge_attr,
                src_key_padding_mask):

        x = x + self._mha_block(x, src_key_padding_mask, edge_attr)
        x = self.norm2(x + self._ff_block(self.norm1(x)))
        return x

    def _mha_block(self,
                   x,
                   src_key_padding_mask,
                   relative_encodings_emb):
        '''
        # Note here that B includes both batch size and time dimension.
        :param x: (N, B, H)
        :param src_key_padding_mask: (B, N)
        :param relative_encodings_emb: (B, N * N, H)
        :return: (B, N, H)
        '''

        M = x.shape[0]
        BS = x.shape[1]

        # Construct the adjacency matrix
        adj = torch.ones((BS, M, M), device=x.device)
        mask = src_key_padding_mask.unsqueeze(-1).repeat(1, 1, M)
        adj.masked_fill_(mask, 0)
        adj.masked_fill_(mask.transpose(1, 2), 0)

        # Compute edge_index
        b, trg, src = torch.where(adj)
        src = src + (b * M)
        trg = trg + (b * M)
        edge_index = torch.stack([src, trg], dim=0)

        # Compute edge_attr
        adj_mask = adj.reshape(-1).bool()
        edge_attr = relative_encodings_emb.reshape(-1, self.d_model)
        edge_attr = edge_attr[adj_mask]
        
        # (B, N, H) --> (B * N, H)
        x = x.permute(1, 0, 2).reshape(-1, x.shape[-1])

        # unvectorized implementation
        # data_list = []
        # for b in range(BS):
        #     adj = torch.ones(M, M).to(x.device)
            
        #     for m in range(M):
        #         if src_key_padding_mask[b, m] == 1:
        #             adj[:, m] = 0
        #             adj[m, :] = 0

        #     # Note that e_ij from edge_attr is information from j to i, this is why src is index 1 and trg is index 0
        #     src = torch.where(adj)[1].unsqueeze(0).to(x.device)
        #     trg = torch.where(adj)[0].unsqueeze(0).to(x.device)
            
        #     edge_index = torch.cat([src, trg], dim=0).long()
        #     mask = adj.reshape(-1).bool()
            
        #     edge_attr = relative_encodings_emb[b]
        #     edge_attr = edge_attr[mask]

        #     data = Data(edge_attr=edge_attr, edge_index=edge_index, x=x[:, b])

        #     data_list.append(data)
        
        # batch = Batch.from_data_list(data_list)
        
        # out_proj is the linear at the end of MHA
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr).reshape(BS, M, self.d_model).permute(1, 0, 2)
        x = self.out_proj(x)
        
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def message(self,
                x_i,
                x_j,
                edge_attr,
                index,
                ptr,
                size_i):
        
        query = self.lin_q_node(x_i).reshape(-1, self.nhead, self.d_model // self.nhead)
        key_node = self.lin_k_node(x_j).reshape(-1, self.nhead, self.d_model // self.nhead)
        value_node = self.lin_v_node(x_j).reshape(-1, self.nhead, self.d_model // self.nhead)

        key_edge = self.lin_k_edge(edge_attr).reshape(-1, self.nhead, self.d_model // self.nhead)
        value_edge = self.lin_v_edge(edge_attr).reshape(-1, self.nhead, self.d_model // self.nhead)
        
        scale = (self.d_model // self.nhead) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        mes = (value_node + value_edge) * alpha.unsqueeze(-1)
        return mes.reshape(-1, self.d_model)

    # inputs is the message
    def update(self,
               inputs,
               x):
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)


class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel):
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class FourierEmbedding(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int):
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
                for _ in range(input_dim)])
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,
                continuous_inputs,
                categorical_embs):
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError('Both continuous_inputs and categorical_embs are None')
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            # Warning: if your data are noisy, don't use learnable sinusoidal embedding
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)


class MapEncoderPtsMA(nn.Module):
    '''
    This class operates on the multi-agent road lanes provided as a tensor with shape
    (B, num_agents, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, cfg):
        super(MapEncoderPtsMA, self).__init__()

        self.cfg_rl_waymo = cfg.dataset.waymo
        self.cfg_model = cfg.model

        self.dropout = self.cfg_model.dropout
        self.d_k = self.cfg_model.hidden_dim
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_attr = self.cfg_rl_waymo.map_attr
        self.num_road_types = self.cfg_rl_waymo.num_road_types

        # Seed parameters for the map
        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)

        # self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(2, self.d_k)))
        self.road_type_lin = nn.Sequential(init_(nn.Linear(8, self.d_k)))
        self.road_pt_type_mlp = nn.Sequential(
            init_(nn.Linear(2*self.d_k, self.d_k * 3)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k * 3, self.d_k)),
        )
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k*3)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k*3, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, :, -1], dim=3) == 0
        road_pts_mask = (1.0 - roads[:, :, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[3])

        # The next lines ensure that we do not obtain NaNs during training for missing agents or for empty roads.
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[3]] = False  # for empty agents
        road_segment_mask[:, :, 0][road_segment_mask.sum(-1) == road_segment_mask.shape[2]] = False  # for empty roads
        return road_segment_mask, road_pts_mask

    def forward(self, roads, road_types):
        '''
        :param roads: (B, M, S, P, k_attr+1)  where B is batch size, M is num_agents, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, M, d_k) where T_obs is the observation horizon. THis tensor is obtained from
        AutoBot's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        M = roads.shape[1]
        S = roads.shape[2]
        P = roads.shape[3]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :, :2]).view(B*M*S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        map_seeds = self.map_seeds.repeat(1, B * M * S, 1)
        road_seg_emb = self.road_pts_attn_layer(query=map_seeds, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, M, S, -1)

        road_type_emb = self.road_type_lin(road_types)
        road_seg_emb = self.road_pt_type_mlp(torch.cat((road_seg_emb, road_type_emb), dim=-1))

        return road_seg_emb.permute(2, 0, 1, 3), road_segment_mask


class DiT(nn.Module):
    def __init__(self, cfg):
        super(DiT, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.cfg_rl_waymo = cfg.dataset.waymo
        self.cfg_model = cfg.model

        self.k_attr = self.cfg_rl_waymo.k_attr
        self.tgt_k_attr = self.k_attr - 2  # removing width and length
        self.map_attr = self.cfg_rl_waymo.map_attr
        self.action_attr = self.cfg_rl_waymo.action_dim
        self.goal_dim = self.cfg_rl_waymo.goal_dim
        self.reward_attr = self.cfg_model.num_reward_components
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
        self.use_rtg = self.cfg_model.use_rtg

        self.embed_state_action = MLPLayer(self.k_attr+self.num_agent_types+self.action_attr, self.d_k, self.d_k)
        self.embed_goal = MLPLayer(self.goal_dim, self.d_k, self.d_k)
        self.embed_timestep = nn.Embedding(self.cfg_rl_waymo.max_timestep, self.d_k)

        if self.use_rtg:
            self.embed_rtg_goal = nn.Embedding(self.cfg_rl_waymo.rtg_discretization, self.d_k)
            self.embed_rtg_veh = nn.Embedding(self.cfg_rl_waymo.rtg_discretization, self.d_k)
            self.embed_rtg_road = nn.Embedding(self.cfg_rl_waymo.rtg_discretization, self.d_k)
            self.embed_rtg = nn.Linear(self.d_k * self.cfg_model.num_reward_components, self.d_k)

            self.embed_all_elements = MLPLayer(self.d_k * 3, self.d_k, self.d_k)  # concat state, goal, rtgs 
        else:
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

        if self.cfg_model.diffusion_type=='states_actions':
            self.output_mlp = MLPLayer(self.d_k, self.d_k, self.tgt_k_attr+self.action_attr)
        elif self.cfg_model.diffusion_type=='actions_only':
            self.output_mlp = MLPLayer(self.d_k, self.d_k, self.action_attr)
        
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

    def forward(self, future_k, cond, diffusion_step, returns=None, eval=False):
        (agent_past_states, agent_past_actions, agent_past_rel_encodings, agent_future_rel_encodings, 
                agent_types, goals, timesteps, rtgs, road_points, road_types, moving_agent_masks, translation_yaws) = cond
        
        in_horizon = agent_past_states.shape[2]
        batch_size, num_agents, pred_horizon, _ = future_k.shape

        agent_types = agent_types.unsqueeze(2).repeat(1, 1, pred_horizon+in_horizon, 1)  # [B, N, Tin+Tout, 5]
        width_length = agent_past_states[:, :, -1:, -3:-1].repeat(1, 1, pred_horizon, 1)  # [B, N, Tout, 2]
        future_state_actions_k = torch.cat((future_k[..., :self.tgt_k_attr], width_length, future_k[..., self.tgt_k_attr:]), dim=-1)  # [B, N, Tout, 9]
        past_state_actions = torch.cat((agent_past_states[..., :-1], agent_past_actions), dim=-1)  # [B, N, Tin, 9]
        agent_past_existence = agent_past_states[..., -1]
        agent_existence = torch.cat((agent_past_existence, agent_past_existence[:, :, -1:].repeat(1, 1, pred_horizon)), dim=-1).permute(0, 2, 1)
        agent_existence = (1.0 - agent_existence).type(torch.BoolTensor).to(future_k.device)

        past_future_tensor = torch.cat((past_state_actions, future_state_actions_k), dim=2)  # [B, N, Tin+Tou, 9]
        past_future_tensor = torch.cat((past_future_tensor, agent_types), dim=-1)

        # Embedding stuff
        state_act_emb = self.embed_state_action(past_future_tensor)
        goal_emb = self.embed_goal(goals).unsqueeze(2).repeat(1, 1, in_horizon+pred_horizon, 1)
        if not eval:
            goal_dropout_mask = torch.rand(size=(batch_size, self.cfg_rl_waymo.max_num_agents, 1, 1), device=goal_emb.device) > self.cfg_model.goal_dropout
            goal_emb = goal_emb * goal_dropout_mask

        timestep_emb = self.embed_timestep(timesteps)[:, :, :, 0]
        if self.use_rtg:
            goal_rtg_emb = self.embed_rtg_goal(rtgs[:, :, -1:, 0])
            veh_rtg_emb = self.embed_rtg_veh(rtgs[:, :, -1:, 1])
            road_rtg_emb = self.embed_rtg_road(rtgs[:, :, -1:, 2])
            rtg_all_emb = self.embed_rtg(torch.cat((goal_rtg_emb, veh_rtg_emb, road_rtg_emb), dim=-1)).repeat(1, 1, pred_horizon+in_horizon, 1)
            agent_embedding = self.embed_all_elements(torch.cat((state_act_emb, goal_emb, rtg_all_emb), dim=-1)) + timestep_emb
        else:
            agent_embedding = self.embed_all_elements(torch.cat((state_act_emb, goal_emb), dim=-1)) + timestep_emb
        agent_embedding = agent_embedding.permute(2, 0, 1, 3)

        diff_step_emb = self.diffustion_step_encoder(self.sincos_emb(diffusion_step))
        diff_step_emb = diff_step_emb[None, :, None]

        #[B, T_obs, M, M, k_attr_rel]
        relative_encodings = torch.cat((agent_past_rel_encodings, agent_future_rel_encodings), dim=3)
        relative_encodings_emb = self.relative_encodings_encoder(relative_encodings).permute(0, 3, 1, 2, 4)
        
        # Process through AutoBot's encoder
        map_features, road_segs_masks = self.map_encoder(road_points, road_types)
        for i in range(self.L_enc):
            agent_embedding = agent_embedding + diff_step_emb
            agent_embedding = self.temporal_attn_fn(agent_embedding, agent_existence, layer=self.temporal_attn_layers[i])
            agent_embedding = self.social_attn_fn(agent_embedding, agent_existence, relative_encodings_emb, layer=self.social_attn_layers[i])
            agent_embedding = self.map_attn_fn(agent_embedding, map_features, road_segs_masks, layer=self.map_attn_layers[i])

        out_states_eps = self.output_mlp(agent_embedding[in_horizon:]).permute(1, 2, 0, 3)
        return out_states_eps

