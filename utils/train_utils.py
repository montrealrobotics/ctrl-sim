import torch.nn as nn
import torch
from torch.nn import Transformer

def create_lambda_lr(cfg):
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else max(
            0.0,
            (cfg.train['max_steps'] - current_step) / (cfg.train['max_steps'] - cfg.train['warmup_steps'])
        )
    )

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def get_causal_mask(cfg, num_timesteps, num_types):
    num_agents = cfg.dataset.waymo.max_num_agents 
    num_steps = num_timesteps
    if cfg.model.decision_transformer:
        state_index = 1
    else:
        state_index = 0
    num_tokens = num_agents * num_steps * num_types
    
    mask = Transformer.generate_square_subsequent_mask(num_tokens)
    multi_agent_mask = torch.Tensor(mask.shape).fill_(0)
    offset = 0
    index = 0
    for index in range(len(multi_agent_mask)):
        mask_out = torch.Tensor(num_agents * num_types).fill_(float('-inf'))
        agent_id = (index // num_types) % num_agents 
        mask_out[agent_id*num_types:(agent_id+1)*(num_types)] = 0
        multi_agent_mask[index, offset:offset+(num_agents * num_types)] = mask_out 
        
        if (index + 1) % (num_agents * num_types) == 0:
            offset += num_agents * num_types

    mask = torch.minimum(mask, multi_agent_mask)

    # current state of all agents is visible
    for index_i in range(len(mask)):
        timestep_idx = index_i // (num_types * num_agents)
        for index_j in range(len(mask)):
            if index_j < (timestep_idx + 1) * (num_agents*num_types) and index_j % num_types == state_index:
                mask[index_i, index_j] = 0.

    
    if cfg.model.attend_own_return_action:
        # mask the actions/returns of other agents at past timesteps (agent should only have access to present/past states of other agents
        # as well as its own actions/returns at present/past timesteps)
        for index_i in range(len(mask)):
            agent_idx_i = (index_i // num_types) % num_agents
            timestep_idx_i = index_i // (num_types * num_agents)
            for index_j in range(len(mask)):
                agent_idx_j = (index_j // num_types) % num_agents
                timestep_idx_j = index_j // (num_types * num_agents)
                type_idx_j = index_j % 3

                if timestep_idx_j < timestep_idx_i:
                    if agent_idx_i != agent_idx_j:
                        if type_idx_j != state_index:
                            mask[index_i, index_j] = float('-inf')

    return mask