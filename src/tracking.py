from utils import config, geometry


def make_tracker():
    return {"tracks": [], "next_id": 0}

def update_tracks(state, detections):
    matched_track_ids = set()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = geometry.get_center(x1, y1, x2, y2)
        det_class = det["class"]

        best_track = None
        best_distance = float("inf")

        for track in state["tracks"]:
            if track["class"] != det_class:
                continue
            if track["id"] in matched_track_ids:
                continue
            tx, ty = track["center"]
            dist = geometry.get_distance(cx, cy, tx, ty)
            if dist < best_distance:
                best_distance = dist
                best_track = track

        if best_track is not None and best_distance < config.MAX_DISTANCE:
            best_track["bbox"] = (x1, y1, x2, y2)
            best_track["center"] = (cx, cy)
            best_track["history"].append((cx, cy))
            best_track["missing_frames"] = 0
            matched_track_ids.add(best_track["id"])
        else:
            state["tracks"].append({
                "id": state["next_id"],
                "class": det_class,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "history": [(cx, cy)],
                "missing_frames": 0
            })
            matched_track_ids.add(state["next_id"])
            state["next_id"] += 1

    for track in state["tracks"][:]:
        if track["id"] not in matched_track_ids:
            track["missing_frames"] += 1
        if track["missing_frames"] > config.MAX_MISSING_FRAME:
            state["tracks"].remove(track)

    return state["tracks"]
