import json
import os
import sys
import time
import argparse
import pickle
import random

import hydra
from policies import AutoregressivePolicy, CTGPlusPlusPolicy
from models import CtRLSim, CTGPlusPlus
from evaluators import PlannerAdversaryEvaluator       
from cfgs.config import CONFIG_PATH   

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    planner_model_path = cfg.eval_planner_adversary.planner.model_path 
    planner_name = cfg.eval_planner_adversary.planner.model
    planner_key_dict = {
        'next_acceleration': 'next_planner_acceleration',
        'next_steering': 'next_planner_steering',
        'rtgs': 'planner_rtgs'
    }
    
    if 'ctrl_sim' in planner_name:
        planner_tilt_dict = {
            'tilt': True,
            'goal_tilt': cfg.eval_planner_adversary.planner.goal_tilt,
            'veh_veh_tilt': cfg.eval_planner_adversary.planner.veh_veh_tilt,
            'veh_edge_tilt': cfg.eval_planner_adversary.planner.veh_edge_tilt
        }
    else:
        planner_tilt_dict = {
            'tilt': False,
            'goal_tilt': None,
            'veh_veh_tilt': None,
            'veh_edge_tilt': None
        }
    
    if planner_name == 'ctg_plus_plus':
        planner_model = CTGPlusPlus.load_from_checkpoint(planner_model_path)
        planner = CTGPlusPlusPolicy(cfg=cfg, 
                                   model_path=planner_model_path,
                                   model=planner_model,
                                   use_rtg=cfg.eval_planner_adversary.planner.use_rtg, 
                                   predict_rtgs=cfg.eval_planner_adversary.planner.predict_rtgs, 
                                   discretize_rtgs=cfg.eval_planner_adversary.planner.discretize_rtgs, 
                                   real_time_rewards=cfg.eval_planner_adversary.planner.real_time_rewards, 
                                   privileged_return=cfg.eval_planner_adversary.planner.privileged_return, 
                                   max_return=cfg.eval_planner_adversary.planner.max_return,
                                   min_return=cfg.eval_planner_adversary.planner.min_return,
                                   key_dict=planner_key_dict, 
                                   tilt_dict=planner_tilt_dict, 
                                   name=planner_name,
                                   sampling_frequency=cfg.eval_planner_adversary.planner.sampling_frequency,
                                   history_steps=cfg.eval_planner_adversary.history_steps)
    else:
        planner_model = CtRLSim.load_from_checkpoint(planner_model_path)
        planner = AutoregressivePolicy(cfg=cfg, 
                                      model_path=planner_model_path,
                                      model=planner_model,
                                      use_rtg=cfg.eval_planner_adversary.planner.use_rtg, 
                                      predict_rtgs=cfg.eval_planner_adversary.planner.predict_rtgs, 
                                      discretize_rtgs=cfg.eval_planner_adversary.planner.discretize_rtgs, 
                                      real_time_rewards=cfg.eval_planner_adversary.planner.real_time_rewards, 
                                      privileged_return=cfg.eval_planner_adversary.planner.privileged_return, 
                                      max_return=cfg.eval_planner_adversary.planner.max_return,
                                      min_return=cfg.eval_planner_adversary.planner.min_return,
                                      key_dict=planner_key_dict, 
                                      tilt_dict=planner_tilt_dict, 
                                      name=planner_name,
                                      action_temperature=cfg.eval_planner_adversary.planner.action_temperature, 
                                      nucleus_sampling=cfg.eval_planner_adversary.planner.nucleus_sampling, 
                                      nucleus_threshold=cfg.eval_planner_adversary.planner.nucleus_threshold)

    adversary_model_path = cfg.eval_planner_adversary.adversary.model_path 
    adversary_name = cfg.eval_planner_adversary.adversary.model
    adversary_key_dict = {
        'next_acceleration': 'next_adversary_acceleration',
        'next_steering': 'next_adversary_steering',
        'rtgs': 'adversary_rtgs'
    }
    
    if 'ctrl_sim' in adversary_name:
        adversary_tilt_dict = {
            'tilt': True,
            'goal_tilt': cfg.eval_planner_adversary.adversary.goal_tilt,
            'veh_veh_tilt': cfg.eval_planner_adversary.adversary.veh_veh_tilt,
            'veh_edge_tilt': cfg.eval_planner_adversary.adversary.veh_edge_tilt
        }
    else:
        adversary_tilt_dict = {
            'tilt': False,
            'goal_tilt': None,
            'veh_veh_tilt': None,
            'veh_edge_tilt': None
        }
    
    if adversary_name == 'ctg_plus_plus':
        adversary_model = CTGPlusPlus.load_from_checkpoint(adversary_model_path)
        adversary = CTGPlusPlusPolicy(cfg=cfg, 
                                   model_path=adversary_model_path,
                                   model=adversary_model,
                                   use_rtg=cfg.eval_planner_adversary.adversary.use_rtg, 
                                   predict_rtgs=cfg.eval_planner_adversary.adversary.predict_rtgs, 
                                   discretize_rtgs=cfg.eval_planner_adversary.adversary.discretize_rtgs, 
                                   real_time_rewards=cfg.eval_planner_adversary.adversary.real_time_rewards, 
                                   privileged_return=cfg.eval_planner_adversary.adversary.privileged_return, 
                                   max_return=cfg.eval_planner_adversary.adversary.max_return,
                                   min_return=cfg.eval_planner_adversary.adversary.min_return,
                                   key_dict=adversary_key_dict, 
                                   tilt_dict=adversary_tilt_dict, 
                                   name=adversary_name,
                                   sampling_frequency=cfg.eval_planner_adversary.adversary.sampling_frequency,
                                   history_steps=cfg.eval_planner_adversary.history_steps)
    else:
        adversary_model = CtRLSim.load_from_checkpoint(adversary_model_path)
        adversary = AutoregressivePolicy(cfg=cfg, 
                                      model_path=adversary_model_path,
                                      model=adversary_model,
                                      use_rtg=cfg.eval_planner_adversary.adversary.use_rtg, 
                                      predict_rtgs=cfg.eval_planner_adversary.adversary.predict_rtgs, 
                                      discretize_rtgs=cfg.eval_planner_adversary.adversary.discretize_rtgs, 
                                      real_time_rewards=cfg.eval_planner_adversary.adversary.real_time_rewards, 
                                      privileged_return=cfg.eval_planner_adversary.adversary.privileged_return, 
                                      max_return=cfg.eval_planner_adversary.adversary.max_return,
                                      min_return=cfg.eval_planner_adversary.adversary.min_return,
                                      key_dict=adversary_key_dict, 
                                      tilt_dict=adversary_tilt_dict, 
                                      name=adversary_name,
                                      action_temperature=cfg.eval_planner_adversary.adversary.action_temperature, 
                                      nucleus_sampling=cfg.eval_planner_adversary.adversary.nucleus_sampling, 
                                      nucleus_threshold=cfg.eval_planner_adversary.adversary.nucleus_threshold)

    evaluator = PlannerAdversaryEvaluator(cfg, planner, adversary)
    metrics_dict, metrics_str = evaluator.evaluate_planner_adversary()
    print(metrics_str)

if __name__ == "__main__":
    main()