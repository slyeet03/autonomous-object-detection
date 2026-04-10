import cv2 as cv
import numpy as np

COLORS = {
    "Smooth Driver":   (50, 200, 80),    # green
    "Moderate Driver": (30, 140, 255),   # orange
    "Risky Driver":    (50, 50, 220),    # red
    "Analyzing":       (160, 160, 160),  # gray
}

FLAG_COLORS = {
    "speeding":        (50,  50,  220),   # red (BGR)
    "aggressive_acc":  (30,  140, 255),   # orange
    "erratic":         (200, 80,  180),   # purple
}
FLAG_OFF_COLOR = (70, 70, 70)
FLAG_KEYS = ["speeding", "aggressive_acc", "erratic"]


def draw(frame, tracks, results, max_speed=1.0):
    if isinstance(results, dict) and results.get("status") == "warming_up":
        cv.putText(frame, "warming up...", (10, 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        return frame

    results_map = {r["id"]: r for r in results}
    speeds_map  = {}
    for r in results:
        speeds_map[r["id"]] = r.get("avg_speed", 0)

    overlay = frame.copy()

    for track in tracks:
        tid  = track["id"]
        x1, y1, x2, y2 = track["bbox"]
        history = track.get("history", [])
        result  = results_map.get(tid)

        if result:
            label     = result["label"]
            score     = result["score"]
            flags     = result.get("flags", {})
            avg_speed = result.get("avg_speed", 0)
        else:
            label     = "Analyzing"
            score     = "-"
            flags     = {}
            avg_speed = 0

        color = COLORS.get(label, COLORS["Analyzing"])

        # history trail 
        if len(history) >= 2:
            pts = np.array(history, dtype=np.int32)
            n   = len(pts)
            for i in range(1, n):
                alpha = i / n                          
                seg_color = tuple(int(c * alpha) for c in color)
                cv.line(overlay, pts[i - 1], pts[i], seg_color, 1, cv.LINE_AA)

        # bounding box 
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # label text 
        text_y = max(y1 - 22, 14)
        cv.putText(frame, f"#{tid} | {label} | {score}",
                   (x1, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv.LINE_AA)

        # speed bar
        bar_x1 = x1
        bar_x2 = x2
        bar_y  = y2 + 4
        bar_h  = 4
        fill   = int((avg_speed / max(max_speed, 1)) * (bar_x2 - bar_x1))
        cv.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 50), -1)
        cv.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill, bar_y + bar_h), color, -1)

        # flag dots 
        dot_y  = bar_y + bar_h + 5
        dot_r  = 4
        dot_x  = x1
        for key in FLAG_KEYS:
            active = flags.get(key, False)
            c = FLAG_COLORS[key] if active else FLAG_OFF_COLOR
            cv.circle(frame, (dot_x + dot_r, dot_y), dot_r, c, -1, cv.LINE_AA)
            dot_x += dot_r * 2 + 3

    # blend trail overlay
    cv.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    return frame
