import torch
import torch.nn as nn

from utils.train_utils import weight_init, get_causal_mask
from utils.layers import MLPLayer
import torch.nn.functional as F 

class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_rl_waymo = self.cfg.dataset.waymo
        self.action_dim = self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.cfg_model.hidden_dim, 
                                                                                    dim_feedforward=self.cfg_model.dim_feedforward,
                                                                                    nhead=self.cfg_model.num_heads,
                                                                                    batch_first=True), 
                                                                                    num_layers=self.cfg_model.num_decoder_layers)
        self.predict_action = MLPLayer(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim, self.action_dim)

        if self.cfg_model.predict_rtg:
            self.predict_rtg = MLPLayer(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim, self.cfg_rl_waymo.rtg_discretization * self.cfg_model.num_reward_components)

        if self.cfg_model.predict_future_states:
            self.predict_future_states = MLPLayer(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim, self.cfg_rl_waymo.train_context_length * 2)

        if not (self.cfg_model.trajeglish or self.cfg_model.il):
            num_types = 3
        elif self.cfg_model.trajeglish:
            num_types = 1
        else:
            num_types = 2
        self.causal_mask = get_causal_mask(self.cfg, self.cfg_rl_waymo.train_context_length, num_types)
        self.apply(weight_init)


    def forward(self, data, scene_enc, eval=False):
        agent_states = data['agent'].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]
        
        # [batch_size, num_timesteps * num_agents * 3, hidden_dim]
        stacked_embeddings = scene_enc['stacked_embeddings']
        # [batch_size, num_polyline_tokens + num_initial_state_tokens, hidden_dim]
        encoder_embeddings = scene_enc['encoder_embeddings']
        # [batch_size, num_polyline_tokens + num_initial_state_tokens]
        src_key_padding_mask = scene_enc['src_key_padding_mask']
        num_timesteps = agent_states.shape[2]
        
        output = self.transformer_decoder(stacked_embeddings, encoder_embeddings, tgt_mask=self.causal_mask.to(stacked_embeddings.device), memory_key_padding_mask=src_key_padding_mask)
        
        preds = {}
        if not (self.cfg_model.trajeglish or self.cfg_model.il):
            # [batch_size, 3, num_timesteps * num_agents, hidden_dim]
            output = output.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, 3, self.cfg_model.hidden_dim).permute(0, 2, 1, 3)
            action_preds = self.predict_action(output[:, 1])
        elif self.cfg_model.trajeglish:
            output = output.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, 1, self.cfg_model.hidden_dim).permute(0, 2, 1, 3)
            action_preds = self.predict_action(output[:, 0])
        else:
            output = output.reshape(batch_size, seq_len*self.cfg_rl_waymo.max_num_agents, 2, self.cfg_model.hidden_dim).permute(0, 2, 1, 3)
            action_preds = self.predict_action(output[:, 0])
        # [batch_size, num_agents, num_timesteps, action_dim]
        action_preds = action_preds.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, self.action_dim).permute(0, 2, 1, 3)
        preds['action_preds'] = action_preds

        if self.cfg_model.predict_future_states:
            state_preds = self.predict_future_states(output[:, 2])
            state_preds = state_preds.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, self.cfg_rl_waymo.train_context_length * 2).permute(0, 2, 1, 3)
            preds['state_preds'] = state_preds
        
        if self.cfg_model.predict_rtg:
            rtg_preds = self.predict_rtg(output[:, 0])
            rtg_preds = rtg_preds.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, self.cfg_rl_waymo.rtg_discretization * self.cfg_model.num_reward_components).permute(0, 2, 1, 3)
            preds['rtg_preds'] = rtg_preds

        return preds