import json
import os
import time
import hydra
import numpy as np
import nocturne
import imageio
import pickle

from nocturne import Simulation
from nocturne.bicycle_model import BicycleModel
from cfgs.config import set_display_window
from utils.data import get_object_type_str, get_road_type_str
from utils.geometry import angle_sub 
from utils.sim import *
from tqdm import tqdm

def apply_adv_traj(veh, t, gt_data_dict, vehicle_data_dict, adv_traj, veh_exists, dt):
    veh_id = veh.getID()
    if not veh_exists:
        acceleration = 0.0
        steering = 0.0
        veh.setPosition(-1000000, -1000000)  # make cars disappear if they are out of actions
    else:
        bike_model = BicycleModel(x=adv_traj[t+1, 0],
                                    y=adv_traj[t+1, 1],
                                    theta=adv_traj[t+1, 4],
                                    vel=np.sqrt(adv_traj[t+1, 2] ** 2 + adv_traj[t+1, 3] ** 2),
                                    L=gt_data_dict[veh_id]['traj'][t+1][-1],
                                    dt=dt)
        
        acceleration, steering, _, _ = bike_model.backward(prev_pos=np.array([veh.getPosition().x,veh.getPosition().y]), 
                                                    prev_theta=veh.getHeading(),
                                                    prev_vel=veh.getSpeed())
    
    veh_action = [acceleration, steering]
    
    if acceleration > 0.0:
        veh.acceleration = acceleration
    else:
        veh.brake(np.abs(acceleration))
    veh.steering = steering

    return veh, veh_action


def apply_gt_action(veh, t, gt_data_dict, vehicle_data_dict, veh_exists, dt):
    veh_id = veh.getID()
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
                                    dt=dt)
        
        acceleration, steering, _, _ = bike_model.backward(prev_pos=np.array([veh.getPosition().x,veh.getPosition().y]), 
                                                    prev_theta=veh.getHeading(),
                                                    prev_vel=veh.getSpeed())
    
    veh_action = [acceleration, steering]
    
    if acceleration > 0.0:
        veh.acceleration = acceleration
    else:
        veh.brake(np.abs(acceleration))
    veh.steering = steering

    return veh, veh_action  


def get_planner_adversary(cfg, gt_data_dict, file, eval_planner_dict, file_to_id, test_filenames):
    def get_veh_id(position_to_match):
        matched_veh_id = None
        for veh_id in gt_data_dict.keys():
            gt_traj = np.array(gt_data_dict[veh_id]["traj"])
            if np.all(np.isclose(gt_traj[0, :2], position_to_match, atol=1e-6)):
                matched_veh_id = veh_id 
                break 
        
        return matched_veh_id

    planner_dict_idx = file_to_id[test_filenames[file]]
    ego_id = eval_planner_dict[planner_dict_idx]['nocturne_sdc_id']
    adversary_id = eval_planner_dict[planner_dict_idx]['nocturne_adversary_id']

    # not valid without an adversary trajectory from CAT (need to evaluate over the same scenes even if we don't use CAT)
    if 'adv_traj' not in eval_planner_dict[planner_dict_idx]:
        return None, None, None
            
    nocturne_waymo_json_file_path = os.path.join(cfg.nocturne_waymo_val_interactive_folder, test_filenames[file])
    with open(nocturne_waymo_json_file_path, 'r') as f:
        json_dict = json.load(f)

    ego_veh_id = get_veh_id(np.array([json_dict['objects'][ego_id]['position'][0]['x'], json_dict['objects'][ego_id]['position'][0]['y']]))
    adversary_veh_id = get_veh_id(np.array([json_dict['objects'][adversary_id]['position'][0]['x'], json_dict['objects'][adversary_id]['position'][0]['y']]))
    
    adv_pos = eval_planner_dict[planner_dict_idx]['adv_traj']
    adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
    adv_vel = get_polyline_vel(adv_pos)
    adv_traj = np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1)
    
    return ego_veh_id, adversary_veh_id, adv_traj


def collect_data(cfg, dt, steps, eval_planner_dict, output_dir):
    test_filenames = []
    file_to_id = {}
    for k in eval_planner_dict:
        filename = eval_planner_dict[k]['nocturne_path'][66:]
        test_filenames.append(filename)
        file_to_id[filename] = k

    num_files_collected = 0
    files = list(np.arange(len(test_filenames)))
    for enum, file in tqdm(enumerate(files)):
        if enum < cfg.cat.start_idx:
            continue
        
        gt_data_dict = get_ground_truth_states(cfg, cfg.nocturne_waymo_val_interactive_folder, test_filenames, file, dt, steps)
        sim = get_sim(cfg, cfg.nocturne_waymo_val_interactive_folder, test_filenames, file)
        scenario = sim.getScenario()
        vehicles = scenario.vehicles()
        for veh in vehicles:
            for veh in vehicles:
                veh.expert_control = False 
                veh.physics_simulated = True

        # adv_traj is the DenseTNT generated adversarial trajectory
        ego_veh_id, adversary_veh_id, adv_traj = get_planner_adversary(cfg, gt_data_dict, file, eval_planner_dict, file_to_id, test_filenames)
        if ego_veh_id is None or adversary_veh_id is None:
            continue
        if num_files_collected > cfg.cat.num_files_to_collect:
            break
        
        ego_vehicle = ego_veh_id 
        adversary_vehicle = adversary_veh_id
        
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
                
                if t >= cfg.nocturne.history_steps - 1 and veh_id == adversary_vehicle:
                    veh, veh_action = apply_adv_traj(veh, t, gt_data_dict, vehicle_data_dict, adv_traj, veh_exists, dt)
                else:
                    veh, veh_action = apply_gt_action(veh, t, gt_data_dict, vehicle_data_dict, veh_exists, dt)
                # Compute reward 
                reward = compute_reward(cfg.nocturne['rew_cfg'], veh, goal_dict[veh_id], goal_dist_normalizer[veh_id], vehicle_data_dict, collision_fix=cfg.nocturne.collision_fix)

                # Append vehicle state data
                vehicle_data_dict[veh_id]["position"].append({'x': veh.getPosition().x, 'y': veh.getPosition().y})
                vehicle_data_dict[veh_id]["velocity"].append({'x': veh.velocity().x, 'y': veh.velocity().y})
                vehicle_data_dict[veh_id]["heading"].append(veh.getHeading())
                vehicle_data_dict[veh_id]["existence"].append(veh_exists)
                vehicle_data_dict[veh_id]["acceleration"].append(veh_action[0])
                vehicle_data_dict[veh_id]["steering"].append(veh_action[1])
                vehicle_data_dict[veh_id]["reward"].append(reward)

            sim.step(dt)

        output_vehicle_data_dict = {}
        attributes = ['position', 'velocity', 'heading', 'existence', 'acceleration', 'steering', 'reward', 'goal_position', 'goal_heading', 'goal_speed', 'width', 'length', 'type']
        for veh_id in vehicle_data_dict.keys():
            output_vehicle_data_dict[veh_id] = {}
            for a in attributes:
                output_vehicle_data_dict[veh_id][a] = vehicle_data_dict[veh_id][a]
        
        file_name = f"{test_filenames[file].split('.')[0]}_physics_cat.json"
        road_data = get_road_data(scenario)
        veh_ids = [v.getID() for v in vehicles]
        focal_agent_idx = veh_ids.index(adversary_vehicle)
        export_data = {"name": file_name, "objects": [*output_vehicle_data_dict.values()], "roads": road_data, 'focal_agent_idx': focal_agent_idx}

        with open(os.path.join(output_dir, file_name), 'w') as file:
            json.dump(export_data, file)
        num_files_collected += 1

        sim.reset()


@hydra.main(version_base=None, config_path="../cfgs/", config_name="config")
def main(cfg):
    
    with open(cfg.cat.dict_path, 'rb') as f:
        eval_planner_dict = pickle.load(f)

    if not os.path.exists(cfg.cat.output_dir):
        os.makedirs(cfg.cat.output_dir, exist_ok=True)
    
    collect_data(cfg=cfg, 
                dt=cfg.nocturne.dt, 
                steps=cfg.nocturne.steps, 
                eval_planner_dict=eval_planner_dict,
                output_dir=cfg.cat.output_dir)

    print("Done!")

if __name__ == '__main__':
    main()
