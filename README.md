# Driver Behaviour Analysis System

A real-time driver behaviour monitoring system that uses computer vision to detect, track, and score vehicles from a traffic camera feed. It analyses how vehicles move and flags dangerous driving patterns like speeding, aggressive acceleration, and erratic movement.

---

## How It Works

The system processes a video feed frame by frame through a pipeline:

```
Video frame → YOLO detection → Centroid tracker → Feature extraction → Behaviour scoring → Annotated output
```

1. **Detection** — A YOLO model detects vehicles in each frame and returns bounding boxes
2. **Tracking** — Vehicles are tracked across frames using centroid distance matching, giving each a persistent ID
3. **Feature Extraction** — For each tracked vehicle, speed, acceleration, path length, displacement, and direction changes are computed from its movement history
4. **Behaviour Analysis** — Each vehicle is scored relative to all other vehicles currently on screen using percentile-based thresholds
5. **Visualization** — Results are drawn onto the frame with colour-coded bounding boxes and labels
6. **Reporting** — When the session ends, a CSV report is saved with every vehicle's final score and flags

---

## Behaviour Flags

| Flag                    | What it means                                                                 |
| ----------------------- | ----------------------------------------------------------------------------- |
| Speeding                | Vehicle's average speed is in the top 85th percentile of all current vehicles |
| Aggressive acceleration | Vehicle's max acceleration is in the top 90th percentile                      |
| Erratic movement        | High path-to-displacement ratio combined with frequent direction changes      |

---

## Scoring

Each vehicle starts with a score of **12** and loses points per flag:

| Penalty                 | Points deducted |
| ----------------------- | --------------- |
| Speeding                | -3              |
| Aggressive acceleration | -4              |
| Erratic movement        | -5              |

### Labels

| Score   | Label           |
| ------- | --------------- |
| 10 – 12 | Smooth Driver   |
| 6 – 9   | Moderate Driver |
| 0 – 5   | Risky Driver    |

> Scoring is **relative**, not absolute — vehicles are compared against each other, not against a fixed speed limit. This makes the system camera-position agnostic.

---

## Output

At the end of every session a `report.csv` is saved in the working directory:

```
vehicle_id, label,            score, speeding, aggressive_acc, erratic
0,          Smooth Driver,    11,    False,    False,          False
1,          Risky Driver,     3,     True,     True,           True
2,          Moderate Driver,  8,     False,    True,           False
```

Logs are printed to the console with timestamps:

```
2024-01-15 10:23:45 | INFO | Report saved to report.csv
2024-01-15 10:23:41 | WARNING | can't receive frame
```

---

## Project Structure

```
├── models/
│   └── best.pt                  # YOLO model weights
├── test_footage/
│   └── drive_test.mp4           # Test video
├── src/
│   ├── main.py                  # Entry point
│   ├── detection.py             # YOLO inference
│   ├── tracking.py              # Centroid-based vehicle tracker
│   ├── features.py              # Feature extraction from track history
│   ├── behaviour.py             # Threshold computation, scoring, labelling
│   ├── visualization.py         # Colour-coded bounding boxes and labels on frames
│   ├── report.py                # Session summary exported to CSV
│   ├── logger.py                # Structured logging with timestamps
│   └── utils/
│       ├── config.py            # All tuneable constants
│       ├── geometry.py          # Distance and center helpers
│       └── helpers.py           # General utilities
```

---

## Configuration

All tuneable parameters live in `src/utils/config.py`:

| Parameter           | Default | Description                                                  |
| ------------------- | ------- | ------------------------------------------------------------ |
| `CONF_THRESHOLD`    | 0.4     | YOLO detection confidence cutoff                             |
| `MAX_DISTANCE`      | 50      | Max pixel distance to match a detection to an existing track |
| `MAX_MISSING_FRAME` | 5       | Frames before a track is dropped                             |
| `FPS`               | 30      | Target frames per second                                     |
| `SPEED_HIGH`        | 0.85    | Percentile threshold for speeding flag                       |
| `ACCEL_HIGH`        | 0.90    | Percentile threshold for aggressive acceleration flag        |
| `PATH_RATIO_HIGH`   | 1.5     | Path-to-displacement ratio above which erratic check runs    |
| `THRESHOLD_ANGLE`   | 0.436   | Minimum direction change angle in radians (~25°)             |
| `SPEED_PENALTY`     | 3       | Score deduction for speeding                                 |
| `ACC_PENALTY`       | 4       | Score deduction for aggressive acceleration                  |
| `ERRATIC_PENALTY`   | 5       | Score deduction for erratic movement                         |

---

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- Ultralytics YOLOv8
- NumPy

Install dependencies:

```bash
pip install opencv-python ultralytics numpy
```

---

## Running

```bash
cd src
python main.py
```

Press `q` to quit. A `report.csv` will be saved automatically when the session ends.

---

## Known Limitations

- Needs at least 3 vehicles on screen before analysis begins (warming up state)
- Speed is measured in pixels per frame, not real-world km/h
- Tracks are not persisted between sessions — each run starts fresh
