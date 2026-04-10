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

latest_frames: dict = {}
frames_lock = threading.Lock()

stop_event = threading.Event()

def process_video(vid_path, report_path):
    model = detection.load_model()
    tracker_state = tracking.make_tracker()

    latest_results = {}
    cached_analysis = None
    recent_features = deque(maxlen=30)  # keep last 30 vehicles worth of features

    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {vid_path}")
        return

    window = vid_path

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"[{vid_path}] finished or can't receive frame")
            break

        detections = detection.predict(model, frame)
        tracks = tracking.update_tracks(tracker_state, detections)
        feature_list = features.extract_features(tracks)

        # add current features into the rolling pool
        for f in feature_list:
            recent_features.append(f)

        # score current vehicles against the rolling pool, not just who's on screen now
        cached_analysis = behaviour.analyze_behaviour(feature_list, list(recent_features))

        if isinstance(cached_analysis, list):
            for r in cached_analysis:
                latest_results[r["id"]] = r

        drawn_frame = visualization.draw(frame, tracks, cached_analysis)

        with frames_lock:
            latest_frames[window] = drawn_frame

    report.generate_report(latest_results, filename=report_path)
    cap.release()

    with frames_lock:
        latest_frames[window] = None


def compute_grid(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def run(video_configs):
    n = len(video_configs)
    rows, cols = compute_grid(n)

    CELL_W, CELL_H = 640, 360
    WIN_W = cols * CELL_W
    WIN_H = rows * CELL_H

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Driver Behaviour Monitor")
    clock = pygame.time.Clock()

    # start one thread per video
    threads = []
    for vid_path, report_path in video_configs:
        t = threading.Thread(
            target=process_video,
            args=(vid_path, report_path),
            daemon=True
        )
        threads.append(t)
        t.start()

    keys = [cfg[0] for cfg in video_configs]  # video paths as keys

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                pygame.quit()
                return
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_q]:
            stop_event.set()
            pygame.quit()
            return

        screen.fill((20, 20, 20))

        with frames_lock:
            current_frames = {k: latest_frames.get(k) for k in keys}

        for i, key in enumerate(keys):
            row = i // cols
            col = i % cols
            x = col * CELL_W
            y = row * CELL_H

            frame = current_frames.get(key)

            if frame is None:
                # if video is finished or have not started yet —> draw a black cell with label
                done_surf = pygame.Surface((CELL_W, CELL_H))
                done_surf.fill((40, 40, 40))
                font = pygame.font.SysFont(None, 28)
                label = font.render(f"{key} — done", True, (180, 180, 180))
                done_surf.blit(label, (10, CELL_H // 2))
                screen.blit(done_surf, (x, y))
            else:
                # convert OpenCV BGR frame → pygame surface
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame_resized = cv.resize(frame_rgb, (CELL_W, CELL_H))
                surf = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
                screen.blit(surf, (x, y))

        pygame.display.flip()
        clock.tick(30)  # cap display at 30fps so main thread doesn't spin

    stop_event.set()
    pygame.quit()


if __name__ == "__main__":
    videos = [
        ("../test_footage/traffic_test.mp4", "../results/report_1.csv"),
        ("../test_footage/drive_test.mp4", "../results/report_2.csv"),
        ("../test_footage/crash.mp4", "../results/report_3.csv"),
        ("../test_footage/crash_snipped.mov", "../results/report_4.csv"),
    ]
    run(videos)
