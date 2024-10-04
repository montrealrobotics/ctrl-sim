import os
import numpy as np
import nocturne

from nocturne import Simulation
from cfgs.config import get_scenario_dict
from utils.data import get_agent_type_onehot, get_road_type_str
from utils.geometry import angle_sub

def get_sim(cfg, files_path, files, file_id):
    """Initialize the scenario."""
    scenario_path = os.path.join(files_path, files[file_id])
    # load scenario, set vehicles to be expert-controlled
    sim = Simulation(scenario_path=scenario_path, config=get_scenario_dict(cfg))
    for obj in sim.getScenario().getObjectsThatMoved():
        obj.expert_control = True

    return sim

def get_ground_truth_states(cfg, files_path, files, file_id, dt, steps, gt_ctg_plus_plus_state=False):
    """retrieves the ground-truth trajectory for first "steps" steps for all vehicles in simulation"""
    def get_state(veh):
        pos = veh.getPosition()
        heading = veh.getHeading()
        target = veh.getGoalPosition()
        speed = veh.getSpeed()
        agent_type = get_agent_type_onehot(veh.getType().value)
        existence = 1 if pos.x != -10000 else 0
        length = veh.getLength()
        
        if gt_ctg_plus_plus_state:
            width = veh.getWidth()
            veh_state = [pos.x, pos.y, speed * np.cos(heading), speed * np.sin(heading), heading, length, width, existence]
        else:
            veh_state = [pos.x, pos.y, heading, speed, existence, target.x, target.y, length]
        veh_type = agent_type

        return veh_state, veh_type
    
    sim = get_sim(cfg, files_path, files, file_id)
    scenario = sim.getScenario()
    vehicles = scenario.vehicles()
    state_dict = {veh.getID(): {"traj": [], "type": None} for veh in vehicles}

    for veh in vehicles:
        veh.expert_control = True
    
    for s in range(steps):
        for veh in vehicles:
            veh_state, veh_type = get_state(veh)
            state_dict[veh.getID()]["traj"].append(veh_state)
            state_dict[veh.getID()]["type"] = veh_type
        
        sim.step(dt)
    
    # we gather all steps+1 timesteps of ground-truth states so that we can compute inverse bicycle model actions
    # as it requires pairs (0,1), (1,2), (2,3), ..., (89, 90)
    for veh in vehicles:
        veh_state, veh_type = get_state(veh)
        state_dict[veh.getID()]["traj"].append(veh_state)
        state_dict[veh.getID()]["type"] = veh_type
    
    # clean up
    sim.reset()
    return state_dict

def get_road_data(scenario):
    road_data = []
    for road_line in scenario.getRoadLines():
        road_type = get_road_type_str(road_line) # {road_line, road_edge, lane, speed_bump, crosswalk}
        geometry = [{"x": pt.x, "y": pt.y} for pt in road_line.geometry_points()]
        road_data.append({"geometry": geometry, "type": road_type})
    
    # Stop signs (static objects)
    for stop_sign in scenario.stop_signs():
        pos = stop_sign.position()
        road_data.append({"geometry": {"x": pos.x, "y": pos.y}, "type": "stop_sign"})

    return road_data

# Compute reward for vehicle at current timestep
# NOTE: Adapted from base_env.py step()
def compute_reward(rew_cfg, veh_obj, goal_dict, goal_dist_normalizer, vehicle_data_dict, collision_fix=True):
    position_target_achieved = True
    speed_target_achieved = True
    heading_target_achieved = True
    obj_pos = veh_obj.position
    obj_pos = np.array([obj_pos.x, obj_pos.y])
    obj_speed = veh_obj.speed 
    obj_heading = veh_obj.heading
    goal_pos = goal_dict['pos']
    goal_speed = goal_dict['speed']
    goal_heading = goal_dict['heading']

    veh_id = veh_obj.getID()

    if rew_cfg['position_target']:
        # if the goal is achieved in a prior timestep, we set goal achieved to True for remainder of trajectory
        if len(vehicle_data_dict[veh_id]['reward']) > 0 and vehicle_data_dict[veh_id]['reward'][-1][0]:
            position_target_achieved = float(True) 
        else:
            position_target_achieved = float(np.linalg.norm(goal_pos - obj_pos) < rew_cfg['position_target_tolerance'])
    if rew_cfg['speed_target']:
        speed_target_achieved = float(np.abs(goal_speed - obj_speed) < rew_cfg['speed_target_tolerance'])
    if rew_cfg['heading_target']:
        heading_target_achieved = float(np.abs(angle_sub(goal_heading, obj_heading)) < rew_cfg['heading_target_tolerance'])

    if rew_cfg['shaped_goal_distance'] and rew_cfg['position_target']:
        # penalize the agent for its distance from goal
        # we scale by goal_dist_normalizers to ensure that this value is always less than the penalty for collision
        goal_dist_scaling = rew_cfg.get('shaped_goal_distance_scaling', 1.0)
        reward_scaling = rew_cfg['reward_scaling']
        if goal_dist_normalizer == 0.0:
            goal_dist_normalizer = 1.0
        
        # if we reach the goal set the shaped reward to maximum possible value
        if len(vehicle_data_dict[veh_id]['reward']) > 0 and vehicle_data_dict[veh_id]['reward'][-1][0]:
            pos_goal_rew = goal_dist_scaling / reward_scaling
        else:
            pos_goal_rew = goal_dist_scaling * (1 - np.linalg.norm(goal_pos - obj_pos) / goal_dist_normalizer) / reward_scaling

        # repeat the same thing for speed and heading
        if rew_cfg['shaped_goal_distance'] and rew_cfg['speed_target']:
           speed_goal_rew = goal_dist_scaling * (1 - np.abs(obj_speed - goal_speed) / 40.0) / reward_scaling

        if rew_cfg['shaped_goal_distance'] and rew_cfg['heading_target']:
            heading_goal_rew = goal_dist_scaling * (1 - np.abs(angle_sub(obj_heading, goal_heading)) / (2 * np.pi)) / reward_scaling
    else:
        pos_goal_rew = 0.0
        speed_goal_rew = 0.0
        heading_goal_rew = 0.0

    if collision_fix:
        veh_veh_collision = float(veh_obj.collision_type_veh == nocturne.CollisionType.VEHICLE_VEHICLE)
        veh_edge_collision = float(veh_obj.collision_type_edge == nocturne.CollisionType.VEHICLE_ROAD)
    else:
        veh_veh_collision = float(veh_obj.collision_type == nocturne.CollisionType.VEHICLE_VEHICLE)
        veh_edge_collision = float(veh_obj.collision_type == nocturne.CollisionType.VEHICLE_ROAD)
    reward = [position_target_achieved, heading_target_achieved, speed_target_achieved, pos_goal_rew,
              speed_goal_rew, heading_goal_rew, veh_veh_collision, veh_edge_collision]
    return reward


def compute_reward_old(rew_cfg, veh_obj, goal_dict, goal_dist_normalizer):
    position_target_achieved = True
    speed_target_achieved = True
    heading_target_achieved = True
    obj_pos = veh_obj.position
    obj_pos = np.array([obj_pos.x, obj_pos.y])
    obj_speed = veh_obj.speed 
    obj_heading = veh_obj.heading
    goal_pos = goal_dict['pos']
    goal_speed = goal_dict['speed']
    goal_heading = goal_dict['heading']

    veh_id = veh_obj.getID()

    if rew_cfg['position_target']:
        # if the goal is achieved in a prior timestep, we set goal achieved to True for remainder of trajectory
        position_target_achieved = float(np.linalg.norm(goal_pos - obj_pos) < rew_cfg['position_target_tolerance'])
    if rew_cfg['speed_target']:
        speed_target_achieved = float(np.abs(goal_speed - obj_speed) < rew_cfg['speed_target_tolerance'])
    if rew_cfg['heading_target']:
        heading_target_achieved = float(np.abs(angle_sub(goal_heading, obj_heading)) < rew_cfg['heading_target_tolerance'])

    if rew_cfg['shaped_goal_distance'] and rew_cfg['position_target']:
        # penalize the agent for its distance from goal
        # we scale by goal_dist_normalizers to ensure that this value is always less than the penalty for collision
        goal_dist_scaling = rew_cfg.get('shaped_goal_distance_scaling', 1.0)
        reward_scaling = rew_cfg['reward_scaling']
        if goal_dist_normalizer == 0.0:
            goal_dist_normalizer = 1.0
        
        # if we reach the goal set the shaped reward to maximum possible value
        pos_goal_rew = goal_dist_scaling * (1 - np.linalg.norm(goal_pos - obj_pos) / goal_dist_normalizer) / reward_scaling

        # repeat the same thing for speed and heading
        if rew_cfg['shaped_goal_distance'] and rew_cfg['speed_target']:
           speed_goal_rew = goal_dist_scaling * (1 - np.abs(obj_speed - goal_speed) / 40.0) / reward_scaling

        if rew_cfg['shaped_goal_distance'] and rew_cfg['heading_target']:
            heading_goal_rew = goal_dist_scaling * (1 - np.abs(angle_sub(obj_heading, goal_heading)) / (2 * np.pi)) / reward_scaling
    else:
        pos_goal_rew = 0.0
        speed_goal_rew = 0.0
        heading_goal_rew = 0.0

    veh_veh_collision = float(veh_obj.collision_type == nocturne.CollisionType.VEHICLE_VEHICLE)
    veh_edge_collision = float(veh_obj.collision_type == nocturne.CollisionType.VEHICLE_ROAD)
    reward = [position_target_achieved, heading_target_achieved, speed_target_achieved, pos_goal_rew,
              speed_goal_rew, heading_goal_rew, veh_veh_collision, veh_edge_collision]
    return reward


def get_moving_vehicles(scenario):
    return [v.getID() for v in scenario.getObjectsThatMoved()]

def moving_average(data, window_size):
    interval = np.pad(data,window_size//2,'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res

def get_polyline_yaw(polyline):
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    diff = polyline_post - polyline
    polyline_yaw = np.arctan2(diff[:,1],diff[:,0])
    polyline_yaw[-1] = polyline_yaw[-2]
    #polyline_yaw = np.where(polyline_yaw<0,polyline_yaw+2*np.pi,polyline_yaw)
    for i in range(len(polyline_yaw)-1):
        if polyline_yaw[i+1] - polyline_yaw[i] > 1.5*np.pi:
            polyline_yaw[i+1] -= 2*np.pi
        elif polyline_yaw[i] - polyline_yaw[i+1] > 1.5*np.pi:
            polyline_yaw[i+1] += 2*np.pi
    return moving_average(polyline_yaw, window_size = 5)

def get_polyline_vel(polyline):
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    diff = polyline_post - polyline
    polyline_vel = diff / 0.1
    return polyline_vel

    