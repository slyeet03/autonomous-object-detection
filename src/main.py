# src/main.py
import queue
import threading
import time
from collections import deque

import cv2 as cv
import numpy as np
import pygame

import behaviour
import detection
import features
import report
import tracking
import visualization
from logger import logger
from utils import config

latest_frames:  dict = {}
latest_results: dict = {}   # key = vid_path, value = dict of id->result
frames_lock  = threading.Lock()
results_lock = threading.Lock()

stop_event = threading.Event()


def process_video(vid_path, report_path):
    model = detection.load_model()
    tracker_state = tracking.make_tracker()
    local_results = {}
    recent_features = deque(maxlen=30)

    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {vid_path}")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"[{vid_path}] finished or can't receive frame")
            break

        detections = detection.predict(model, frame)
        tracks = tracking.update_tracks(tracker_state, detections)
        feature_list = features.extract_features(tracks)

        for f in feature_list:
            recent_features.append(f)

        analysis = behaviour.analyze_behaviour(feature_list, list(recent_features))

        if isinstance(analysis, list):
            for r in analysis:
                local_results[r["id"]] = r

        # compute max speed across visible vehicles for the speed bar
        max_spd = max((r.get("avg_speed", 0) for r in local_results.values()), default=1)

        drawn_frame = visualization.draw(frame, tracks, analysis, max_speed=max_spd)

        with frames_lock:
            latest_frames[vid_path] = drawn_frame

        with results_lock:
            latest_results[vid_path] = dict(local_results)

    report.generate_report(local_results, filename=report_path)
    cap.release()

    with frames_lock:
        latest_frames[vid_path] = None


def compute_grid(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


SMOOTH_COL   = (50,  200,  80)
MODERATE_COL = (255, 140,  30)
RISKY_COL    = (220,  50,  50)
GRAY_COL     = (130, 130, 130)
BG_COL       = (20,  20,  20)
CELL_BG      = (30,  30,  30)
STRIP_BG     = (18,  18,  18)

FLAG_DOT_COLORS = {
    "speeding":       (220,  50,  50),
    "aggressive_acc": (255, 140,  30),
    "erratic":        (180,  80, 200),
}


def label_color(label):
    if label == "Smooth Driver":   return SMOOTH_COL
    if label == "Moderate Driver": return MODERATE_COL
    if label == "Risky Driver":    return RISKY_COL
    return GRAY_COL


def draw_stats_strip(surface, x, y, w, h, cam_results, font_sm):
    pygame.draw.rect(surface, STRIP_BG, (x, y, w, h))

    counts = {"Smooth Driver": 0, "Moderate Driver": 0, "Risky Driver": 0}
    total  = len(cam_results)
    for r in cam_results.values():
        lbl = r.get("label", "")
        if lbl in counts:
            counts[lbl] += 1

    pad = 8
    cx = x + pad
    items = [
        (f"seen {total}", GRAY_COL),
        (f"smooth {counts['Smooth Driver']}",     SMOOTH_COL),
        (f"mod {counts['Moderate Driver']}",       MODERATE_COL),
        (f"risky {counts['Risky Driver']}",        RISKY_COL),
    ]
    for text, color in items:
        surf = font_sm.render(text, True, color)
        surface.blit(surf, (cx, y + (h - surf.get_height()) // 2))
        cx += surf.get_width() + 14


def draw_leaderboard(surface, x, y, w, h, all_results, font_sm, font_tiny):
    pygame.draw.rect(surface, (15, 15, 15), (x, y, w, h))

    # gather all vehicles, sort by score asc (worst first)
    combined = []
    for cam_key, cam_results in all_results.items():
        cam_name = cam_key.split("/")[-1].split(".")[0]   
        for vid, r in cam_results.items():
            combined.append((r["score"], r, cam_name))
    combined.sort(key=lambda t: t[0])

    title = font_tiny.render("RISKIEST VEHICLES", True, (80, 80, 80))
    surface.blit(title, (x + 10, y + (h - title.get_height()) // 2))
    cx = x + 10 + title.get_width() + 16

    for score, r, cam_name in combined[:6]:
        label    = r.get("label", "")
        col      = label_color(label)
        flags    = r.get("flags", {})
        tag_text = f"#{r['id']} {cam_name}  score {score}"
        tag_surf = font_sm.render(tag_text, True, col)

        tag_w = tag_surf.get_width() + 36
        tag_h = h - 10
        tag_y = y + 5

        pygame.draw.rect(surface, (35, 35, 35), (cx, tag_y, tag_w, tag_h), border_radius=4)
        surface.blit(tag_surf, (cx + 6, tag_y + (tag_h - tag_surf.get_height()) // 2))

        # three flag dots
        dot_x = cx + tag_surf.get_width() + 10
        dot_y = tag_y + tag_h // 2
        for key in ["speeding", "aggressive_acc", "erratic"]:
            active = flags.get(key, False)
            dot_col = FLAG_DOT_COLORS[key] if active else (55, 55, 55)
            pygame.draw.circle(surface, dot_col, (dot_x, dot_y), 4)
            dot_x += 11

        cx += tag_w + 8
        if cx > x + w - 60:
            break


CELL_W, CELL_H  = 640, 360
STRIP_H         = 24   # per-cell stats strip height
LEADERBOARD_H   = 36   # bottom bar height

def run(video_configs):
    n = len(video_configs)
    rows, cols = compute_grid(n)

    WIN_W = cols * CELL_W
    WIN_H = rows * (CELL_H + STRIP_H) + LEADERBOARD_H

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Driver Behaviour Monitor")
    clock = pygame.time.Clock()

    font_sm   = pygame.font.SysFont("monospace", 13)
    font_tiny = pygame.font.SysFont("monospace", 11)

    threads = []
    for vid_path, report_path in video_configs:
        t = threading.Thread(target=process_video, args=(vid_path, report_path), daemon=True)
        threads.append(t)
        t.start()

    keys = [cfg[0] for cfg in video_configs]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                pygame.quit()
                return
        if pygame.key.get_pressed()[pygame.K_q]:
            stop_event.set()
            pygame.quit()
            return

        screen.fill(BG_COL)

        with frames_lock:
            current_frames = {k: latest_frames.get(k) for k in keys}
        with results_lock:
            current_results = {k: dict(latest_results.get(k, {})) for k in keys}

        for i, key in enumerate(keys):
            row = i // cols
            col = i % cols
            x   = col * CELL_W
            y   = row * (CELL_H + STRIP_H)

            frame = current_frames.get(key)

            if frame is None:
                done_surf = pygame.Surface((CELL_W, CELL_H))
                done_surf.fill(CELL_BG)
                lbl = font_sm.render(f"{key} — done", True, (120, 120, 120))
                done_surf.blit(lbl, (10, CELL_H // 2))
                screen.blit(done_surf, (x, y))
            else:
                frame_rgb     = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame_resized = cv.resize(frame_rgb, (CELL_W, CELL_H))
                surf = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
                screen.blit(surf, (x, y))

            # per-cell stats strip
            cam_res = current_results.get(key, {})
            draw_stats_strip(screen, x, y + CELL_H, CELL_W, STRIP_H, cam_res, font_sm)

        # global leaderboard bar
        draw_leaderboard(screen, 0, rows * (CELL_H + STRIP_H),
                         WIN_W, LEADERBOARD_H, current_results, font_sm, font_tiny)

        pygame.display.flip()
        clock.tick(30)

    stop_event.set()
    pygame.quit()


if __name__ == "__main__":
    videos = [
        ("../test_footage/test1.mp4", "../results/report_1.csv"),
        ("../test_footage/test2.mp4", "../results/report_2.csv"),
        ("../test_footage/test3.mp4", "../results/report_3.csv"),
        ("../test_footage/test4.mov", "../results/report_4.csv"),
        ("../test_footage/test5.mp4", "../results/report_5.csv"),
        ("../test_footage/test6.mp4", "../results/report_6.csv"),

    ]
    run(videos)
