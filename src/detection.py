from ultralytics import YOLO

from utils import config


def load_model(model_path=config.MODEL_PATH):
    return YOLO(model_path)

def predict(model, frame):
    detections = []
    results = model(frame, conf=config.CONF_THRESHOLD, verbose=False)
    r = results[0]
    boxes = r.boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class": model.names[cls],
                "confidence": conf
            })

    return detections
