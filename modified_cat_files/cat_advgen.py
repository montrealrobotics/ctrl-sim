import argparse
import numpy as np
from tqdm import trange
import time
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from advgen.adv_generator import AdvGenerator
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int,default=32)
    parser.add_argument('--AV_traj_num', type=int,default=1)
    parser.add_argument(
        "--dict_path", default="/scratch/eval_planner_dict.pkl", type=str, help="Path to eval planner dictionary."
    )
    adv_generator = AdvGenerator(parser)

    args = parser.parse_args()

    extra_args = dict(mode="top_down", film_size=(2200, 2200))

    env = WaymoEnv(
            {
                "agent_policy": ReplayEgoCarPolicy,
                "reactive_traffic": False,
                "use_render": False,
                "data_directory": '/scratch/valid_md_scenarios',
                "num_scenarios": 4577,
                "force_reuse_object_name" :True,
                "sequential_seed": True,
                "vehicle_config":dict(show_navi_mark=False,show_dest_mark=False,)
            }
        )

    attack_cnt = 0
    time_cost = 0.

    pbar = trange(4577)
    for i in pbar:
      # These scenarios raise an error when running through MetaDrive, so we skip them
      bad_ids = [92, 335, 482, 616, 965, 1316, 1379, 1756, 1876, 2494, 2949, 3356, 3562, 4081, 4236]
      if i in bad_ids:
        continue
      
      ################ Second Round : create the adversarial counterpart #####################

      with open(args.dict_path, 'rb') as f:
        eval_planner_dict = pickle.load(f)
      
      env.reset(force_seed=i)
      env.vehicle.ego_crash_flag = False
      done = False
      ep_timestep = 0

      t0 = time.time()

      adv_generator.before_episode(env)   # initialization before each episode
      adv_generator.generate()            # Adversarial scenario generation with the logged history corresponding to the current env 
      # generate the adversarial cat trajectory
      adv_traj = np.array(adv_generator.adv_traj)[:, :2]
      eval_planner_dict[i]['adv_traj'] = adv_traj

      # save at end of each iteration in case MetaDrive crashes.
      with open(args.dict_path, 'wb') as f:
        pickle.dump(eval_planner_dict, f)
      
      t1 = time.time()
      time_cost += t1 - t0

    env.close()