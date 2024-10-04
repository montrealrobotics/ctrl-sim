import json
import os
import time
import hydra
import numpy as np
import nocturne
import imageio

from nocturne import Simulation
from nocturne.bicycle_model import BicycleModel
from cfgs.config import set_display_window
from utils.data import get_object_type_str, get_road_type_str
from utils.geometry import angle_sub 
from utils.sim import get_sim, get_ground_truth_states, get_road_data, compute_reward
from tqdm import tqdm

def collect_data(cfg, dt, steps, output_path, files_path, files, chunk):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # loop through all training files
    for file in tqdm(chunk):
        gt_data_dict = get_ground_truth_states(cfg, files_path, files, file, dt, steps)
        sim = get_sim(cfg, files_path, files, file)
        scenario = sim.getScenario()
        vehicles = scenario.vehicles()
        for veh in vehicles:
            if veh.getID() in gt_data_dict.keys():
                veh.expert_control = False
                veh.physics_simulated = True
            else:
                veh.expert_control = True
                veh.physics_simulated = False
        
        # Collect vehicle data
        vehicle_data_dict = {}
        goal_dict = {}
        goal_dist_normalizer = {}

        for t in range(steps):
            for veh in vehicles:
                veh_id = veh.getID()
                
                if veh_id not in gt_data_dict.keys():
                    continue

                if t == 0:
                    goal_pos = np.array([veh.target_position.x, veh.target_position.y])
                    goal_heading = veh.target_heading
                    goal_speed = veh.target_speed
                    gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
                    idx_disappear = np.where(gt_traj_data[:, 4] == 0)[0]
                    if len(idx_disappear) > 0:
                        idx_disappear = idx_disappear[0] - 1
                        if np.linalg.norm(gt_traj_data[idx_disappear, :2] - goal_pos) > 0.0:
                            goal_pos = gt_traj_data[idx_disappear, :2]
                            goal_heading = gt_traj_data[idx_disappear, 2]
                            goal_speed = gt_traj_data[idx_disappear, 3]
                    
                    vehicle_data_dict[veh_id] = {
                        "position": [], # [{'x': float, 'y': float}, ...]
                        "velocity": [],  # [{'x': float, 'y': float}, ...]
                        "heading": [], # [float, ...]
                        "existence": [],
                        "acceleration": [], # [float, ...]
                        "steering": [], # [float, ...]
                        "reward": [], # [float, ...]
                        "goal_position": {'x': goal_pos[0], 'y': goal_pos[1]},
                        "goal_heading": goal_heading,
                        "goal_speed": goal_speed,
                        "width": veh.getWidth(),
                        "length": veh.getLength(),
                        "type": get_object_type_str(veh)
                    }

                    goal_dict[veh_id] = {
                            'pos': goal_pos,
                            'heading': goal_heading,
                            'speed': goal_speed
                        }

                    # Precompute goal-dist normalizer (used for reward computation)
                    obj_pos = veh.getPosition()
                    obj_pos = np.array([obj_pos.x, obj_pos.y])
                    dist = np.linalg.norm(obj_pos - goal_pos)
                    goal_dist_normalizer[veh_id] = dist
                
                # action is only defined if state at next timestep is defined
                veh_exists = gt_data_dict[veh_id]['traj'][t][4] and gt_data_dict[veh_id]['traj'][t+1][4]
                # once we encounter the first missing timestep, all future timesteps are also missing
                # this is because we need contiguous sequence to push through nocturne simulator
                if t > 0 and vehicle_data_dict[veh_id]["existence"][-1] == 0:
                    veh_exists = 0
                
                if not veh_exists:
                    acceleration = 0.0
                    steering = 0.0
                    veh.setPosition(-1000000, -1000000)  # make cars disappear if they are out of actions
                else:
                    bike_model = BicycleModel(x=gt_data_dict[veh_id]['traj'][t+1][0],
                                                y=gt_data_dict[veh_id]['traj'][t+1][1],
                                                theta=gt_data_dict[veh_id]['traj'][t+1][2],
                                                vel=gt_data_dict[veh_id]['traj'][t+1][3],
                                                L=gt_data_dict[veh_id]['traj'][t+1][-1],
                                                dt=0.1)
                    
                    accel, steer, _, _ = bike_model.backward(prev_pos=np.array([veh.getPosition().x,veh.getPosition().y]), 
                                                                prev_theta=veh.getHeading(),
                                                                prev_vel=veh.getSpeed())
                    veh_action = [accel, steer]

                    acceleration = veh_action[0]
                    steering = veh_action[1]

                if acceleration > 0.0:
                    veh.acceleration = acceleration
                else:
                    veh.brake(np.abs(acceleration))
                veh.steering = steering

                # Compute reward 
                reward = compute_reward(cfg.nocturne['rew_cfg'], veh, goal_dict[veh_id], goal_dist_normalizer[veh_id], vehicle_data_dict, collision_fix=cfg.nocturne.collision_fix)

                # Append vehicle state data
                vehicle_data_dict[veh_id]["position"].append({'x': veh.getPosition().x, 'y': veh.getPosition().y})
                vehicle_data_dict[veh_id]["velocity"].append({'x': veh.velocity().x, 'y': veh.velocity().y})
                vehicle_data_dict[veh_id]["heading"].append(veh.getHeading())
                vehicle_data_dict[veh_id]["existence"].append(veh_exists)
                vehicle_data_dict[veh_id]["acceleration"].append(acceleration)
                vehicle_data_dict[veh_id]["steering"].append(steering)
                vehicle_data_dict[veh_id]["reward"].append(reward)

            sim.step(dt)

        road_data = get_road_data(scenario)

        # Save data to files
        file_name = f"{files[file].split('.')[0]}_physics.json"
        export_data = {"name": file_name, "objects": [*vehicle_data_dict.values()], "roads": road_data}

        with open(os.path.join(output_path, file_name), 'w') as file:
            json.dump(export_data, file)

        sim.reset()


@hydra.main(version_base=None, config_path="../cfgs/", config_name="config")
def main(cfg):
    
    if cfg.offline_rl.mode == 'train':
        files_path = cfg.nocturne_waymo_train_folder
        output_path = cfg.offline_rl.output_data_folder_train
    elif cfg.offline_rl.mode == 'val':
        files_path = cfg.nocturne_waymo_val_folder
        output_path = cfg.offline_rl.output_data_folder_val
    else:
        files_path = cfg.nocturne_waymo_val_interactive_folder 
        output_path = cfg.offline_rl.output_data_folder_val_interactive
    
    with open(os.path.join(files_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
        # sort the files so that we have a consistent order
        files = sorted(files)

    chunk = list(range(cfg.offline_rl.chunk_idx * cfg.offline_rl.chunk_size, (cfg.offline_rl.chunk_idx + 1) * cfg.offline_rl.chunk_size))
    if len(files) < chunk[0]:
        raise ValueError("chunk_idx is too large for dataset size.")
    elif len(files) < chunk[-1]:
        chunk = [c for c in chunk if c < len(files)]

    collect_data(cfg=cfg, 
                dt=cfg.nocturne.dt, 
                steps=cfg.nocturne.steps, 
                output_path=output_path,
                files_path=files_path, 
                files=files,
                chunk=chunk)

if __name__ == '__main__':
    main()
