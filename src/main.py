import cv2 as cv

import behaviour
import detection
import features
import tracking
from utils import config


def captureVideo():
    vid_path = "../test_footage/drive_test.mp4"

    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        print("cannot open camera")
        return
    
    cap.set(cv.CAP_PROP_FPS, config.FPS)
    
    actual_fps = cap.get(cv.CAP_PROP_FPS)
    print(f"Desired FPS: {config.FPS}, Actual FPS set by camera: {actual_fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("can't receive frame")
            break

        detections = detection.predict(frame)
        tracks = tracking.update_tracks(detections)
        feature = features.extract_features(tracks)
        results = behaviour.analyze_behaviour(feature)

        print(results)



        cv.imshow("live feed", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


captureVideo()
