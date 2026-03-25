from utils import config, geometry

tracks = []
NEXT_TRACK_ID = 0

def update_tracks(detections):
    global NEXT_TRACK_ID
    matched_track_ids = set()

    # matching detection to existing tracks
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = geometry.get_center(x1, y1, x2, y2)
        det_class = det["class"]

        best_track = None
        best_distance = float("inf")

        for track in tracks:
            # skip different classes
            if track["class"] != det_class:
                continue

            # skip already matched tracks
            if track["id"] in matched_track_ids:
                continue

            # center of last frame
            tx, ty = track["center"]
            # distance b/w last frame center and current frame center
            dist = geometry.get_distance(cx, cy, tx, ty)

            if dist < best_distance:
                best_distance = dist
                best_track = track

        if best_track is not None and best_distance < config.MAX_DISTANCE:
            # update existing track
            best_track["bbox"] = (x1, y1, x2, y2)
            best_track["center"] = (cx, cy)
            best_track["history"].append((cx, cy))
            best_track["missing_frames"] = 0

            matched_track_ids.add(best_track["id"])

        else:
            # create new track
            tracks.append({
                "id": NEXT_TRACK_ID,
                "class": det_class,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "history": [(cx, cy)],
                "missing_frames": 0
            })
            matched_track_ids.add(NEXT_TRACK_ID)
            NEXT_TRACK_ID += 1

    for track in tracks[:]:
        if track["id"] not in matched_track_ids:
            track["missing_frames"] += 1

        if track["missing_frames"] > config.MAX_MISSING_FRAME:
            tracks.remove(track)

    return tracks
