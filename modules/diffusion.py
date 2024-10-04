import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.diffusion_helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
    Progress,
    Silent
)
from modules.ctg_arch import DiT
from modules.diffusion_guidance import n_step_guided_p_sample, GoalGuide, CollisionGuide


class GaussianDiffusion(nn.Module):
    # def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
    #     loss_type='l1', clip_denoised=False, predict_epsilon=True,
    #     action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
    #     condition_guidance_w=0.1):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = DiT(cfg)
        self.horizon = cfg.dataset.waymo.train_context_length - cfg.dataset.waymo.input_horizon
        self.observation_dim = cfg.dataset.waymo.k_attr
        self.observation_dim_nosize = self.observation_dim - 2
        self.action_dim = cfg.dataset.waymo.action_dim
        self.transition_dim = self.action_dim + self.observation_dim_nosize
        self.returns_condition = cfg.model.returns_condition
        self.condition_guidance_w = cfg.model.condition_guidance_w
        n_timesteps = cfg.model.n_diffusion_steps

        # TODO: make use_guidance a class variable.
        # if self.cfg.eval.use_guidance:
        #     self.goal_guide = GoalGuide()
        #     self.collision_guide = CollisionGuide()

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = False
        self.predict_epsilon = cfg.model.predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        action_weight = cfg.model.action_weight
        loss_discount = cfg.model.loss_discount
        loss_weights = None
        loss_type = cfg.train.loss_type
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights = loss_weights[None]

        ## manually set a0 weight
        loss_weights[:, 0, -self.action_dim:] = action_weight  # first timestep action is most important
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        epsilon = self.model(x, cond, t, returns=returns, eval=True)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.cfg.model.n_eval_diffusion_step) if verbose else Silent()
        eval_every = self.n_timesteps // self.cfg.model.n_eval_diffusion_step

        for i in reversed(range(0, self.n_timesteps, eval_every)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # x = self.p_sample(x, cond, timesteps, returns)

            # TODO: make guidance variables class variables. These are policy parameters, so really should be defined in policy configs, not eval
            # if self.cfg.eval.use_guidance:
            #     x, _ = n_step_guided_p_sample(self, x, cond, timesteps, guide=self.collision_guide, scale=self.cfg.eval.guide_scale, 
            #                                   n_guide_steps=self.cfg.eval.n_guidance_steps)
            # else:
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        agent_past_states = cond[0]
        batch_size, num_agents, _, _ = agent_past_states.shape
        shape = (batch_size, num_agents, self.horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None, x_existence=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, cond, t, returns, eval=False)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, x_existence)
        else:
            loss, info = self.loss_fn(x_recon, x_start, x_existence)

        return loss, info

    def combine_states_actions(self, x_tuple):
        x_states, x_actions = x_tuple
        x = torch.cat((x_states[..., :-1], x_actions), dim=-1)
        x_existence = x_states[..., -1]
        return x, x_existence

    def loss(self, x_tuple, cond, returns=None):
        x, x_existences = self.combine_states_actions(x_tuple)
        if self.cfg.model.supervise_moving:
            moving_agent_masks = cond[-2]
            x_existences = x_existences * moving_agent_masks.unsqueeze(-1)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns, x_existences)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
