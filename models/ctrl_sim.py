import os 
import numpy as np 
import random 
from matplotlib import pyplot as plt 
from modules.encoder import Encoder
from modules.decoder import Decoder
from utils.train_utils import create_lambda_lr
from utils.geometry import apply_se2_transform

import torch 
import torch.distributions as D 
import torch.nn.functional as F 
from torch import optim, nn 
from torch.optim.lr_scheduler import LambdaLR, ConstantLR
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
torch.set_printoptions(sci_mode=False)

class CtRLSim(pl.LightningModule):
    def __init__(self, cfg, data_module=None):
        super(CtRLSim, self).__init__()
        
        if not cfg.train.finetuning:
            self.save_hyperparameters()
        self.cfg = cfg 
        self.cfg_model = self.cfg.model
        self.cfg_rl_waymo = self.cfg.dataset.waymo
        self.action_dim = self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization
        self.seq_len = self.cfg_rl_waymo.train_context_length
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.data_module = data_module

    def on_train_epoch_start(self):
        if self.cfg.train.finetuning:
            # Re-sample real dataset indices at the start of each training epoch
            self.data_module.sample_real_indices()
            print("Resampling real data indices")


    def forward(self, data, eval=False):
        scene_enc = self.encoder(data, eval)
        pred = self.decoder(data, scene_enc, eval)

        return pred 

    
    def compute_loss(self, data, preds):
        loss_dict = {}
        if self.cfg_model.trajeglish:
            logits = preds['action_preds']
            logits = logits[:, :, :-1, :]
            logits = logits.reshape(-1, self.cfg_rl_waymo.max_num_agents*(self.seq_len-1), self.action_dim)
            B, T, C = logits.shape 
            # existence mask is whether the next token exists (because if next token exists, current token exists by definition)
            existence_mask = data['agent'].agent_states[:, :, 1:, -1:].reshape(B, T, 1)

            if self.cfg_model.supervise_moving:
                moving_mask = data['agent'].moving_agent_mask.unsqueeze(-1).repeat(1, 1, self.seq_len-1).reshape(B, T, 1)
                existence_mask = moving_mask * existence_mask 

            logits = logits.view(B * T, C)
            actions = data['agent'].actions[:, :, 1:].reshape(-1)
            existence_mask = existence_mask.view(-1)
            loss_actions = F.cross_entropy(logits.float(), actions.long(), reduction='none')
            loss_actions = loss_actions * existence_mask.float()
            loss_actions = (self.cfg_model.loss_action_coef * loss_actions.sum()) / existence_mask.sum()

        else:
            logits = preds['action_preds']
            logits = logits.reshape(-1, self.cfg_rl_waymo.max_num_agents*self.seq_len, self.action_dim)
            B, T, C = logits.shape 
            existence_mask = data['agent'].agent_states[:, :, :, -1:].reshape(B, T, 1)
            
            if self.cfg_model.supervise_moving:
                moving_mask = data['agent'].moving_agent_mask.unsqueeze(-1).repeat(1, 1, self.seq_len).reshape(B, T, 1)
                existence_mask = moving_mask * existence_mask # mask out non-moving agents
            
            logits = logits.view(B * T, C)
            actions = data['agent'].actions.view(-1)
            existence_mask = existence_mask.view(-1)
            loss_actions = F.cross_entropy(logits.float(), actions.long(), reduction='none')
            loss_actions = loss_actions * existence_mask.float()
            loss_actions = (self.cfg_model.loss_action_coef * loss_actions.sum()) / existence_mask.sum()

        loss_dict['loss_actions'] = loss_actions
        
        if self.cfg_model.predict_rtg:
            rtg_preds = preds['rtg_preds'].reshape(-1, self.cfg_rl_waymo.rtg_discretization, self.cfg_model.num_reward_components)
            rtg_goal_logits = rtg_preds[:, :, 0]
            rtg_veh_logits = rtg_preds[:, :, 1]
            rtg_goal = data['agent'].rtgs[:, :, :, 0].reshape(-1)
            rtg_veh = data['agent'].rtgs[:, :, :, 1].reshape(-1)
            
            loss_rtg_goal = F.cross_entropy(rtg_goal_logits.float(), rtg_goal.long(), reduction='none')
            loss_rtg_goal = loss_rtg_goal * existence_mask.float()
            loss_rtg_goal = loss_rtg_goal.sum() / existence_mask.sum()

            loss_rtg_veh = F.cross_entropy(rtg_veh_logits.float(), rtg_veh.long(), reduction='none')
            loss_rtg_veh = loss_rtg_veh * existence_mask.float()
            loss_rtg_veh = loss_rtg_veh.sum() / existence_mask.sum()

            rtg_road_logits = rtg_preds[:, :, 2]
            rtg_road = data['agent'].rtgs[:, :, :, 2].reshape(-1)
            loss_rtg_road = F.cross_entropy(rtg_road_logits.float(), rtg_road.long(), reduction='none')
            loss_rtg_road = loss_rtg_road * existence_mask.float()
            loss_rtg_road = loss_rtg_road.sum() / existence_mask.sum()

            loss_dict['loss_rtg_goal'] = loss_rtg_goal 
            loss_dict['loss_rtg_veh'] = loss_rtg_veh
            loss_dict['loss_rtg_road'] = loss_rtg_road

        
        if self.cfg_model.predict_future_states and not self.cfg_model.local_frame_predictions:
            # [batch_size, num_agents, num_timesteps, 1]
            existence_mask = data['agent'].agent_states[:, :, :, -1:]
            if self.cfg_model.supervise_moving:
                moving_mask = data['agent'].moving_agent_mask.unsqueeze(-1).repeat(1, 1, self.seq_len).unsqueeze(-1)
                existence_mask = moving_mask * existence_mask # mask out non-moving agents
            # [batch_size, num_agents, num_timesteps, 2]
            states = data['agent'].agent_states[:, :, :, :2]
            
            # [1,2,3,4]
            # [2,3,4,mask]
            # [3,4,mask,mask]
            # [4,mask,mask,mask]
            existence_mask_new = torch.zeros_like(existence_mask.unsqueeze(-3).repeat(1, 1, self.cfg_rl_waymo.train_context_length, 1, 1))
            states_new = torch.zeros_like(states.unsqueeze(-3).repeat(1, 1, self.cfg_rl_waymo.train_context_length, 1, 1))
            for i in range(self.cfg_rl_waymo.train_context_length):
                # i = 0: [..., 0, :31] = [..., 1:]
                # i = 1: [..., 1, :30] = [..., 2:]
                existence_mask_new[:, :, i, :(self.cfg_rl_waymo.train_context_length - i - 1)] = existence_mask[:, :, (i+1):]
                # i = 0: [..., 0, 31:] = False
                # i = 1: [..., 1, 30:] = False
                existence_mask_new[:, :, i, (self.cfg_rl_waymo.train_context_length - i - 1):] = False
                # i = 0: [..., 0, :31] = [..., 1:]
                # i = 1: [..., 1, :30] = [..., 2:]
                states_new[:, :, i, :(self.cfg_rl_waymo.train_context_length - i - 1)] = states[:, :, (i+1):]
            
            # [batch_size, num_agents, num_timesteps, num_timesteps * 2] --> [-1, 2]
            state_preds = preds['state_preds'].reshape(-1, 2)
            states = states_new.reshape(-1, 2)
            existence_mask = existence_mask_new
            loss_state = F.mse_loss(state_preds.float(), states.float(), reduction='none').sum(-1)
            loss_state = loss_state * existence_mask.reshape(-1)
            loss_state = loss_state.sum() / (100 * (existence_mask.sum() * 2))

            loss_dict['loss_state'] = loss_state
        
        # local frame predictions
        elif self.cfg_model.predict_future_states:
            existence_mask = data['agent'].agent_states[:, :, :, -1:]
            # position_x, position_y, velocity_x, velocity_y, heading
            states = data['agent'].agent_states[:, :, :, :5]

            existence_mask_new = torch.zeros_like(existence_mask.unsqueeze(-3).repeat(1, 1, self.cfg_rl_waymo.train_context_length, 1, 1))
            states_new = torch.zeros_like(states.unsqueeze(-3).repeat(1, 1, self.cfg_rl_waymo.train_context_length, 1, 1))
            states_present = torch.clone(states)
            for i in range(self.cfg_rl_waymo.train_context_length):
                existence_mask_new[:, :, i, :(self.cfg_rl_waymo.train_context_length - i - 1)] = existence_mask[:, :, (i+1):]
                existence_mask_new[:, :, i, (self.cfg_rl_waymo.train_context_length - i - 1):] = False
                states_new[:, :, i, :(self.cfg_rl_waymo.train_context_length - i - 1)] = states[:, :, (i+1):]
            
            states = states_new 
            origin_states = states_present
            batch_size, num_agents, num_timesteps, state_dim = origin_states.size()
            origin_expanded = origin_states.unsqueeze(3).expand(-1, -1, -1, num_timesteps, -1)
            translated_states = states - origin_expanded
            yaws = origin_states[..., 4]
            cos_yaws = torch.cos(-yaws).unsqueeze(-1)
            sin_yaws = torch.sin(-yaws).unsqueeze(-1)
            rotation_matrices = torch.stack((cos_yaws, -sin_yaws, sin_yaws, cos_yaws), dim=-1)
            rotation_matrices = rotation_matrices.view(batch_size, num_agents, num_timesteps, 2, 2)
            for i in range(num_timesteps):
                pos = translated_states[:, :, i, :, :2].clone().unsqueeze(-1)
                rot = rotation_matrices[:, :, i].unsqueeze(2)
                rotated_positions = torch.matmul(rot, pos)
                states[:, :, i, :, :2] = rotated_positions.squeeze(-1)

            state_preds = preds['state_preds'].reshape(-1, 2)
            states = states[:, :, :, :, :2].reshape(-1, 2)
            existence_mask = existence_mask_new
            loss_state = F.mse_loss(state_preds.float(), states.float(), reduction='none').sum(-1)
            loss_state = loss_state * existence_mask.reshape(-1)
            loss_state = loss_state.sum() / (100 * (existence_mask.sum() * 2))

            loss_dict['loss_state'] = loss_state

        return loss_dict


    def training_step(self, data, batch_idx):
        # loss during training is the cross-entropy loss
        preds = self(data)
        loss_dict = self.compute_loss(data, preds)

        self.log('loss', loss_dict['loss_actions'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        if self.cfg_model.predict_rtg:
            self.log('loss_rtg_goal', loss_dict['loss_rtg_goal'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('loss_rtg_veh', loss_dict['loss_rtg_veh'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('loss_rtg_road', loss_dict['loss_rtg_road'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        if self.cfg_model.predict_future_states:
            self.log('loss_state', loss_dict['loss_state'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        final_loss = loss_dict['loss_actions']
        if self.cfg_model.predict_rtg:
            final_loss = final_loss + loss_dict['loss_rtg_goal'] + loss_dict['loss_rtg_veh'] + loss_dict['loss_rtg_road']
        
        if self.cfg_model.predict_future_states:
            final_loss = final_loss + loss_dict['loss_state']
        
        return final_loss


    def validation_step(self, data, batch_idx):
        preds = self(data, eval=True)
        loss_dict = self.compute_loss(data, preds)
        B = preds['action_preds'].shape[0]

        self.log('val_loss', loss_dict['loss_actions'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        if self.cfg_model.predict_rtg:
            self.log('val_rtg_goal_loss', loss_dict['loss_rtg_goal'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=B) 
            self.log('val_rtg_veh_loss', loss_dict['loss_rtg_veh'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
            self.log('val_rtg_road_loss', loss_dict['loss_rtg_road'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)   
        if self.cfg_model.predict_future_states:
            self.log('val_state_loss', loss_dict['loss_state'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)   


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms_encoder = grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms_encoder)

        norms_decoder = grad_norm(self.decoder, norm_type=2)
        self.log_dict(norms_decoder)


    ### Taken largely from QCNet repository
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        
        if self.cfg.train.finetuning:
            optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr(self.cfg))
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}



