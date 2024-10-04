import torch
import torch.nn as nn
from utils.diffusion_helpers import extract

'''
Set of functions to be used for the CTG++ baseline.
We really want three functions:
1. minimize distance to goal.
2. maximize distance to other agents.
3. maxmize distance to nearest road edge.
'''

@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.01, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=False,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond)

        if scale_grad_by_std:
            grad = model_var * grad
            
        grad[t < t_stopgrad] = 0

        x = x + scale * grad

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y


class GoalGuide(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, cond):
        # output = self.model(x, cond, t)
        (agent_past_states, agent_past_actions, agent_past_rel_encodings, agent_future_rel_encodings, 
                agent_types, goals, timesteps, rtgs, road_points, road_types, moving_agent_masks, translation_yaws) = cond
        existence = agent_past_states[:, :, -1, -1]
        output = torch.norm(x[:, :, :, :-2] - goals.unsqueeze(2), p=2, dim=-1).min(-1)[0] * existence
        return -output.squeeze(dim=-1)

    def gradients(self, x, cond):
        x.requires_grad_()
        y = self(x, cond)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad
    
class CollisionGuide(nn.Module):

    def __init__(self):
        super().__init__()

    def make_2d_rotation_matrix(self, angle_in_radians):
        """ Makes rotation matrix to rotate point in x-y plane counterclockwise by angle_in_radians.
        """
        B, N = angle_in_radians.shape
        rotation_tensor = torch.zeros((B, N, 2, 2))
        rotation_tensor[:, :, 0, :2] = torch.cat((torch.cos(angle_in_radians).unsqueeze(-1), 
                                                  -torch.sin(angle_in_radians).unsqueeze(-1)), dim=-1)
        rotation_tensor[:, :, 1, :2] = torch.cat((torch.sin(angle_in_radians).unsqueeze(-1), 
                                                  torch.cos(angle_in_radians).unsqueeze(-1)), dim=-1)
        return rotation_tensor

    def apply_se2_transform(self, coordinates, translation, yaw):
        """
        Converts global coordinates to coordinates in the frame given by the rotation quaternion and
        centered at the translation vector. The rotation is meant to be a z-axis rotation.
        """
        transform = self.make_2d_rotation_matrix(angle_in_radians=yaw).to(coordinates.device)
        rotated_coords = torch.einsum('bnij, bnijk -> bnik', coordinates[:, :, :, :2], transform.unsqueeze(2).repeat(1, 1, 22, 1, 1))
        coordinates = rotated_coords + translation.unsqueeze(2)
        return coordinates

    def forward(self, x, cond):
        # output = self.model(x, cond, t)
        (agent_past_states, agent_past_actions, agent_past_rel_encodings, agent_future_rel_encodings, 
                agent_types, goals, timesteps, rtgs, road_points, road_types, moving_agent_masks, translation_yaws) = cond
        x = self.apply_se2_transform(x, translation=translation_yaws[:, :, :2], yaw=translation_yaws[:, :, -1])
        existence = agent_past_states[:, :, -1, -1].clone()
        existence[existence==0] = 10000
        output = torch.norm(x.unsqueeze(3) - x.unsqueeze(2), p=2, dim=-1) * existence.unsqueeze(-1).unsqueeze(-1)
        diagonal_mask = torch.eye(22).bool()
        output[:, :, diagonal_mask] = 10000
        output = output.min(-1)[0].min(-1)[0] * agent_past_states[:, :, -1, -1]
        return output

    def gradients(self, x, cond):
        x.requires_grad_()
        y = self(x, cond)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad