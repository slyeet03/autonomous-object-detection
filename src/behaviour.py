from logger import logger
from utils import config


def compute_thresholds(features):
    if len(features) < 3:
        return None

    speeds = []
    accelerations = []
    
    for feature in features:
        speeds.append(feature["avg_speed"])
        accelerations.append(feature["acceleration"])

    speeds.sort()
    accelerations.sort()

    speed_idx = int(config.SPEED_HIGH * len(speeds))
    acc_idx = int(config.ACCEL_HIGH * len(accelerations))

    logger.debug(f"speed index: {speed_idx}")
    logger.debug(f"acc index: {acc_idx}")

    if speed_idx >= len(speeds):
        speed_idx = len(speeds) - 1

    if acc_idx >= len(accelerations):
        acc_idx = len(accelerations) - 1

    if not speeds or not accelerations:
        return 0
    else:
        speed_high = speeds[speed_idx]
        acc_high = accelerations[acc_idx]

        thresholds = {}
        thresholds["speed"] = speed_high
        thresholds["acc"] = acc_high

        return thresholds

    
def detect_behaviours(feature, thresholds):
    flag = {}
    flag_count = 0

    avg_speed = feature["avg_speed"]
    acceleration = feature["acceleration"]
    direction_change_count = feature["direction_change_count"]
    path_length = feature["path_length"]
    displacement = feature["displacement"]
        
    if displacement < 1e-6:
        path_ratio = 1.0 #stationary vehicle
    else:
        path_ratio = path_length/displacement

    speed_high = False
    acc_high = False
    erratic = False

    if avg_speed > thresholds["speed"]:
        speed_high = True
        flag_count += 1
        
    if acceleration > thresholds["acc"]:
        acc_high = True
        flag_count += 1

    if path_ratio > config.PATH_RATIO_HIGH:
        if direction_change_count <= config.LOW_DIR_CHANGE:
            erratic = False
        elif direction_change_count >= config.HIGH_DIR_CHANGE:
            erratic = True
            flag_count += 1
        
    flag["speeding"] = speed_high
    flag["aggressive_acc"] = acc_high
    flag["erratic"] = erratic


    return flag, flag_count

def compute_score(flag):
    score = 12

    speeding = flag["speeding"]
    aggressive_acc = flag["aggressive_acc"]
    erratic = flag["erratic"]

    if speeding:
        score -= config.SPEED_PENALTY
    
    if aggressive_acc:
        score -= config.ACC_PENALTY

    if erratic:
        score -= config.ERRATIC_PENALTY

    return score

def assign_label(score):
    label = ""

    if score >= 10:
        label = "Smooth Driver"
    elif score >= 6:
        label = "Moderate Driver"
    else:
        label = "Risky Driver"

    return label

def analyze_behaviour(features, pool=None):
    threshold_source = pool if pool else features
    
    thresholds = compute_thresholds(threshold_source)
    if thresholds is None:
        return {"status": "warming_up"}

    results = []
    for feature in features:
        flag, flag_count = detect_behaviours(feature, thresholds)
        score = compute_score(flag)
        label = assign_label(score)
        results.append({
            "id":       feature["id"],
            "score":    score,
            "label":    label,
            "behaviors": flag_count,
            "flags":    flag,
            "avg_speed": feature["avg_speed"],
        })

    return results
