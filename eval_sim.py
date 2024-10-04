import hydra
from policies import AutoregressivePolicy, CTGPlusPlusPolicy
from models import CtRLSim, CTGPlusPlus
from evaluators import PolicyEvaluator
from cfgs.config import CONFIG_PATH

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):

    model_path = cfg.eval.policy.model_path 
    name = cfg.eval.policy.model
    
    key_dict = {
        'next_acceleration': 'next_acceleration',
        'next_steering': 'next_steering',
        'rtgs': 'rtgs'
    }
    
    if 'ctrl_sim' in name:
        tilt_dict = {
            'tilt': True,
            'goal_tilt': cfg.eval.policy.goal_tilt,
            'veh_veh_tilt': cfg.eval.policy.veh_veh_tilt,
            'veh_edge_tilt': cfg.eval.policy.veh_edge_tilt
        }
    else:
        tilt_dict = {
            'tilt': False,
            'goal_tilt': None,
            'veh_veh_tilt': None,
            'veh_edge_tilt': None
        }
    
    if name == 'ctg_plus_plus':
        model = CTGPlusPlus.load_from_checkpoint(model_path)
        policy = CTGPlusPlusPolicy(cfg=cfg, 
                                   model_path=model_path,
                                   model=model,
                                   use_rtg=cfg.eval.policy.use_rtg, 
                                   predict_rtgs=cfg.eval.policy.predict_rtgs, 
                                   discretize_rtgs=cfg.eval.policy.discretize_rtgs, 
                                   real_time_rewards=cfg.eval.policy.real_time_rewards, 
                                   privileged_return=cfg.eval.policy.privileged_return, 
                                   max_return=cfg.eval.policy.max_return,
                                   min_return=cfg.eval.policy.min_return,
                                   key_dict=key_dict, 
                                   tilt_dict=tilt_dict, 
                                   name=name,
                                   sampling_frequency=cfg.eval.policy.sampling_frequency,
                                   history_steps=cfg.eval.history_steps)
    else:
        model = CtRLSim.load_from_checkpoint(model_path)
        policy = AutoregressivePolicy(cfg=cfg, 
                                      model_path=model_path,
                                      model=model,
                                      use_rtg=cfg.eval.policy.use_rtg, 
                                      predict_rtgs=cfg.eval.policy.predict_rtgs, 
                                      discretize_rtgs=cfg.eval.policy.discretize_rtgs, 
                                      real_time_rewards=cfg.eval.policy.real_time_rewards, 
                                      privileged_return=cfg.eval.policy.privileged_return, 
                                      max_return=cfg.eval.policy.max_return,
                                      min_return=cfg.eval.policy.min_return,
                                      key_dict=key_dict, 
                                      tilt_dict=tilt_dict, 
                                      name=name,
                                      action_temperature=cfg.eval.policy.action_temperature, 
                                      nucleus_sampling=cfg.eval.policy.nucleus_sampling, 
                                      nucleus_threshold=cfg.eval.policy.nucleus_threshold)

    evaluator = PolicyEvaluator(cfg, policy)
    metrics_dict, metrics_str = evaluator.evaluate_policy()
    print(metrics_str)

if __name__ == "__main__":
    main()