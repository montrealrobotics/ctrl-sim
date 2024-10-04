import matplotlib.pyplot as plt 
import numpy as np 
import os 
import json
import math
from tqdm import tqdm 
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import pickle
import random
import subprocess
from moviepy.editor import ImageSequenceClip
from utils.data import *
from utils.geometry import *


def generate_video_frames(data, name, output_dir, dpi):
    max_num_road_pts_per_polyline = 100
    
    png_dir = f'{output_dir}/{name}'
    if not os.path.exists(png_dir):
        os.makedirs(png_dir, exist_ok=True)

    roads_data = data['roads']
    num_roads = len(roads_data)
    final_roads = []
    final_road_types = []
    for n in range(num_roads):
        curr_road_rawdat = roads_data[n]['geometry']
        if isinstance(curr_road_rawdat, dict):
            # for stop sign, repeat x/y coordinate along the point dimension
            final_roads.append(np.array((curr_road_rawdat['x'], curr_road_rawdat['y'], 1.0)).reshape(1, -1).repeat(max_num_road_pts_per_polyline, 0))
            final_road_types.append(get_road_type_onehot(roads_data[n]['type']))
        else:
            # either we add points until we run out of points and append zeros
            # or we fill up with points until we reach max limit
            curr_road = []
            for p in range(len(curr_road_rawdat)):
                curr_road.append(np.array((curr_road_rawdat[p]['x'], curr_road_rawdat[p]['y'], 1.0)))
                if len(curr_road) == max_num_road_pts_per_polyline:
                    final_roads.append(np.array(curr_road))
                    curr_road = []
                    final_road_types.append(get_road_type_onehot(roads_data[n]['type']))
            if len(curr_road) < max_num_road_pts_per_polyline and len(curr_road) > 0:
                tmp_curr_road = np.zeros((max_num_road_pts_per_polyline, 3))
                tmp_curr_road[:len(curr_road)] = np.array(curr_road)
                final_roads.append(tmp_curr_road)
                final_road_types.append(get_road_type_onehot(roads_data[n]['type']))

    final_roads = np.array(final_roads)
    final_road_types = np.array(final_road_types)

    agents_data = data['objects']
    num_agents = len(agents_data)
    agent_data = []
    agent_types = []
    agent_goals = []
    agent_rewards = []
    parked_agent_ids = [] # fade these out
    for n in range(len(agents_data)):
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]
        ag_position = np.column_stack((x_values, y_values))
        ag_heading = np.array(agents_data[n]['heading']).reshape((-1, 1))
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        if np.linalg.norm(ag_velocity, axis=-1).mean() < 0.05:
            parked_agent_ids.append(n)
        ag_existence = np.array(agents_data[n]['existence']).reshape((-1, 1))

        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        agent_type = get_object_type_onehot(agents_data[n]['type'])

        rewards = np.array(agents_data[n]['reward']) * ag_existence

        goal_position_x = agents_data[n]['goal_position']['x']
        goal_position_y = agents_data[n]['goal_position']['y']
        goal_position = np.repeat(np.array([goal_position_x, goal_position_y])[None, :], len(ag_position), 0)

        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
        agent_goals.append(goal_position)
        agent_rewards.append(rewards)
    
    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    agent_goals = np.array(agent_goals)
    agent_rewards = np.array(agent_rewards)
    parked_agent_ids = np.array(parked_agent_ids)

    final_road_points = final_roads
    agent_states = agent_data
    goals = agent_goals
    
    agent_color = '#ffde8b'
    agent_alpha = 0.5
    agent_zord = 2
    
    last_timestep = 0
    for agent_idx in range(len(agent_data)):
        last_timestep_agent = int(np.sum(agent_data[agent_idx, :, -1]) - 1)
        if last_timestep_agent > last_timestep:
            last_timestep = last_timestep_agent 
    
    coordinates = agent_states[:, :, :2]
    coordinates_mask = agent_states[:, :, -1].astype(bool).copy()
    
    x_min_all = 100000
    y_min_all = 100000
    x_max_all = -100000
    y_max_all = -100000
    for a in range(len(coordinates)):
        x_min = np.min(coordinates[a, :, 0][coordinates_mask[a]]) - 25
        x_max = np.max(coordinates[a, :, 0][coordinates_mask[a]]) + 25
        y_min = np.min(coordinates[a, :, 1][coordinates_mask[a]]) - 25
        y_max = np.max(coordinates[a, :, 1][coordinates_mask[a]]) + 25
        if x_min < x_min_all:
            x_min_all = x_min 
        if y_min < y_min_all:
            y_min_all = y_min 
        if x_max > x_max_all:
            x_max_all = x_max
        if y_max > y_max_all:
            y_max_all = y_max

    x_min = x_min_all 
    y_min = y_min_all 
    x_max = x_max_all 
    y_max = y_max_all

    if (x_max - x_min) > (y_max - y_min):
        diff = (x_max - x_min) - (y_max - y_min)
        diff_side = diff / 2
        y_min -= diff_side 
        y_max += diff_side 
    else:
        diff = (y_max - y_min) - (x_max - x_min)
        diff_side = diff / 2
        x_min -= diff_side 
        x_max += diff_side 
    
    # iterate over all video frames
    timesteps = list(range(0, last_timestep + 1))
    
    for e, t in tqdm(enumerate(timesteps)):
        # plot the underlying HD-Map
        for r in range(len(final_road_points)):
            if final_road_types[r, 3] != 1:
                continue
            mask = final_road_points[r, :, 2].astype(bool)
            plt.plot(final_road_points[r, :, 0][mask], final_road_points[r, :, 1][mask], color='grey', linewidth=0.5)
        
        for r in range(len(final_road_points)):
            if final_road_types[r, 2] != 1 and final_road_types[r, 2] != 1:
                continue
            mask = final_road_points[r, :, 2].astype(bool)
            plt.plot(final_road_points[r, :, 0][mask], final_road_points[r, :, 1][mask], color='lightgray', linewidth=0.3)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 

        for a in range(len(coordinates)):
            color = agent_color
            alpha = agent_alpha
            zord = agent_zord
            edgecolor = 'black'
            label = None

            # draw bounding boxes
            length = agent_states[a, t, 5] * 0.8
            width = agent_states[a, t, 6] * 0.8
            bbox_x_min = coordinates[a, t, 0] - width / 2
            bbox_y_min = coordinates[a, t, 1] - length / 2
            lw = (0.35) / ((x_max - x_min) / 140)
            rectangle = mpatches.FancyBboxPatch((bbox_x_min, bbox_y_min),
                                        width, length, ec=edgecolor, fc=color, linewidth=lw, alpha=alpha,
                                        boxstyle=mpatches.BoxStyle("Round", pad=0.3), zorder=4, label=label)
            
            tr = transforms.Affine2D().rotate_deg_around(coordinates[a, t, 0], coordinates[a, t, 1], radians_to_degrees(agent_states[a, t, 4]) - 90) + plt.gca().transData

            # Apply the transformation to the rectangle
            rectangle.set_transform(tr)
            
            plt.gca().set_aspect('equal', adjustable='box')
            # Add the patch to the Axes
            plt.gca().add_patch(rectangle)
            
            heading_length = length / 2 + 1.5
            heading_angle_rad = agent_states[a, t, 4]
            vehicle_center = coordinates[a, t, :2]

            # Calculate end point of the heading line
            line_end_x = vehicle_center[0] + heading_length * math.cos(heading_angle_rad)
            line_end_y = vehicle_center[1] + heading_length * math.sin(heading_angle_rad)

            # Draw the heading line
            plt.plot([vehicle_center[0], line_end_x], [vehicle_center[1], line_end_y], color='black', zorder=6, alpha=0.25, linewidth=0.25 / ((x_max - x_min) / 140))

        plt.tight_layout()
        plt.savefig('{}/frame_{:03}.png'.format(png_dir, e), dpi=dpi, bbox_inches='tight')
        plt.clf()


def generate_video(data, name, output_dir, dpi=100, delete_images=True):
    generate_video_frames(data, name, output_dir, dpi)

    image_folder = f'{output_dir}/{name}'
    
    # Get list of all image files in the directory
    images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    images = [str1.replace('\n', '') for str1 in images]
    images.sort()  # Sort by filename

    # Create a video clip from the image sequence
    clip = ImageSequenceClip(images, fps=10)
    
    # Write the video file
    clip.write_videofile(f"{image_folder}.mp4", codec='libx264')

    if delete_images:
        for image in images:
            os.remove(image)
