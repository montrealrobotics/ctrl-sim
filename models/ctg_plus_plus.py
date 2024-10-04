import os 
import numpy as np 
import random 
from matplotlib import pyplot as plt 
from utils.train_utils import create_lambda_lr
from utils.geometry import apply_se2_transform
from modules.diffusion import GaussianDiffusion
from modules.rtg_model import RTGModel

import torch 
import torch.distributions as D 
import torch.nn.functional as F 
from torch import optim, nn 
from torch.optim.lr_scheduler import LambdaLR, ConstantLR
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
torch.set_printoptions(sci_mode=False)

class CTGPlusPlus(pl.LightningModule):
    def __init__(self, cfg):
        super(CTGPlusPlus, self).__init__()
        
        self.save_hyperparameters()
        self.cfg = cfg 
        self.cfg_model = self.cfg.model
        self.cfg_rl_waymo = self.cfg.dataset.waymo
        self.observational_dim = 7
        self.seq_len = self.cfg_rl_waymo.train_context_length
        self.diff_model = GaussianDiffusion(self.cfg)
        
        if not self.cfg_model.use_rtg:
            self.cfg_model.predict_rtg = False  # hard set to false.
        if self.cfg_model.predict_rtg:
            self.rtg_model = RTGModel(self.cfg)
        
    def extract_data(self, data):
        agent_past_states = data['agent']['agent_past_states'].float()
        agent_past_actions = data['agent']['agent_past_actions'].float()
        agent_future_states = data['agent']['agent_future_states'].float()
        agent_future_actions = data['agent']['agent_future_actions'].float()
        agent_past_rel_encodings = data['agent']['past_relative_encodings'].float()
        agent_future_rel_encodings = data['agent']['future_relative_encodings'].float()
        agent_types = data['agent']['agent_types'].float()
        goals = data['agent']['goals'].float()
        timesteps = data['agent']['timesteps'].long()
        rtgs = data['agent']['rtgs'].long()
        road_points = data['map']['road_points'].float()
        road_types = data['map']['road_types'].float()
        moving_agents = data['agent']['moving_agent_mask'].float()
        translation_yaws = data['agent']['agent_translation_yaws'].float()
        return (agent_past_states, agent_past_actions, agent_past_rel_encodings, agent_future_rel_encodings, 
                agent_types, goals, timesteps, rtgs, road_points, road_types, moving_agents, translation_yaws), \
            (agent_future_states, agent_future_actions)


    def forward(self, data):
        cond_data, diff_data = self.extract_data(data)
        samples = self.diff_model.forward(cond=cond_data)
        return samples

    
    def training_step(self, data, batch_idx):
        cond_data, diff_data = self.extract_data(data)
        loss, info = self.diff_model.loss(x_tuple=diff_data, cond=cond_data)
        self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        if self.cfg_model.predict_rtg:
            loss_rtg_goal, loss_rtg_veh, loss_rtg_road = self.rtg_model.loss(cond_data)
            self.log('rtg_goal_loss', loss_rtg_goal, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('rtg_veh_loss', loss_rtg_veh, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('rtg_road_loss', loss_rtg_road, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        
            return loss + loss_rtg_goal + loss_rtg_veh + loss_rtg_road
        else: 
            return loss

    def validation_step(self, data, batch_idx):
        cond_data, diff_data = self.extract_data(data)
        loss, info = self.diff_model.loss(x_tuple=diff_data, cond=cond_data)
        if self.cfg_model.predict_rtg:
            loss_rtg_goal, loss_rtg_veh, loss_rtg_road = self.rtg_model.loss(cond_data)
        samples = self.diff_model.forward(cond=cond_data)
        gt_future_states = diff_data[0]
        gt_future_actions = diff_data[1]
        exist_mask = gt_future_states[:, :, :, -1]
        if self.cfg_model.supervise_moving:
            moving_agent_mask = cond_data[-2]
            exist_mask = exist_mask * moving_agent_mask.unsqueeze(-1)

        action_mse = torch.linalg.norm(gt_future_actions - samples[:, :, :, -2:], dim=-1) * exist_mask
        action_mse = action_mse.sum() / exist_mask.sum()
        
        pred_states_normalized = samples[:, :, :, :2] * self.cfg_rl_waymo.state_normalizer.pos_div
        gt_states_normalized = gt_future_states[:, :, :, :2] * self.cfg_rl_waymo.state_normalizer.pos_div
        state_mse = torch.linalg.norm(gt_states_normalized - pred_states_normalized, dim=-1) * exist_mask
        state_mse = state_mse.sum() / exist_mask.sum()

        self.log('action_mse', action_mse.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0])
        self.log('state_mse', state_mse.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0])
        self.log('val_action_loss', info['a0_loss'].item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0])
        self.log('val_loss', loss.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0])
        if self.cfg_model.predict_rtg:
            self.log('val_rtg_goal_loss', loss_rtg_goal, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0]) 
            self.log('val_rtg_veh_loss', loss_rtg_veh, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0])
            self.log('val_rtg_road_loss', loss_rtg_road, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=diff_data[0].shape[0])   

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms_encoder = grad_norm(self.diff_model.model, norm_type=2)
        self.log_dict(norms_encoder)


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
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}



