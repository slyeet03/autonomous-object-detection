import math

import numpy as np

from utils import config, geometry

features = []
time = 1 / config.FPS

def calc_speed(history, speeds):
    for i in range(len(history)-2):
        x1 = history[i][0]
        y1 = history[i][1]
        x2 = history[i+1][0]
        y2 = history[i+1][1]

        dist = geometry.get_distance(x1, y1, x2, y2)
            
        speed = dist/time
        speeds.append(speed)

    if not speeds:
        return 0
    
    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
   
    return avg_speed, max_speed

def calc_acc(speeds, accelerations):
    for i in range(len(speeds)-2):
            acceleration = (speeds[i+1] - speeds[i]) / time
            accelerations.append(acceleration)

    abs_acc = [abs(x) for x in accelerations]

    if not abs_acc:
        return 0
    
    max_acc = max(abs_acc)
    return max_acc


def calc_dir_change(history):
    theta = []
    del_theta = []
    count = 0

    for i in range(len(history)-2): 
        x1 = history[i][0]
        y1 = history[i][1]
        x2 = history[i+1][0]
        y2 = history[i+1][1]
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)  
        theta.append(angle)
    
    for i in range(len(theta)): 
        delta = theta[i] - theta[i-1]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        del_theta.append(delta)
    
    for theta in del_theta:
        if theta > config.THRESHOLD_ANGLE:
            count+=1
        
    return count


def calc_path_length(history):
    path_length = 0

    for i in range(len(history)-2):
        x1 = history[i][0]
        y1 = history[i][1]
        x2 = history[i+1][0]
        y2 = history[i+1][1]

        dist = geometry.get_distance(x1, y1, x2, y2)
        path_length += dist

    return path_length

def calc_displacement(history):
    last_idx = len(history)-1
    x1 = history[0][0]
    y1 = history[0][1]
    x2 = history[last_idx][0]
    y2 = history[last_idx][1]

    displacement = geometry.get_distance(x1, y1, x2, y2)

    return displacement 

def extract_features(tracks):
    for track in tracks:
        history = track["history"]
        speeds = []
        accelerations = []
        direction_change_count = 0
        path_length = 0
        
        # skip if the history has less that two data of speed
        if len(history)< 3:
            continue

        avg_speed, max_speed = calc_speed(history, speeds)
        max_acc = calc_acc(speeds, accelerations)
        path_length = calc_path_length(history)
        direction_change_count = calc_dir_change(history)
        displacement = calc_displacement(history)

        features.append({
            "id": track["id"],
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "acceleration": max_acc,
            "direction_change_count": direction_change_count,
            "path_length": path_length,
            "displacement": displacement
        })

    return features
            

