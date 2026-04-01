import cv2 as cv

import behaviour
import detection
import features
import report
import tracking
import visualization
from logger import logger
from utils import config


def captureVideo():
    vid_path = "../test_footage/traffic_test.mp4"

    latest_results = {}

    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        logger.info("cannot open camera")
        return
    
    cap.set(cv.CAP_PROP_FPS, config.FPS)
    
    actual_fps = cap.get(cv.CAP_PROP_FPS)
    print(f"Desired FPS: {config.FPS}, Actual FPS set by camera: {actual_fps}")

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, actual_fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("can't receive frame")
            break

        detections = detection.predict(frame)
        tracks = tracking.update_tracks(detections)
        feature = features.extract_features(tracks)
        results = behaviour.analyze_behaviour(feature)

        if isinstance(results, list):
            for r in results:
                latest_results[r["id"]] = r

        drawn_frame = visualization.draw(frame, tracks, results)

        out.write(drawn_frame)

        cv.imshow("live feed", drawn_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    report.generate_report(latest_results)

    cap.release()
    out.release()
    cv.destroyAllWindows()

captureVideo()
