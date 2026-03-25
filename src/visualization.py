import cv2 as cv


def draw(frame, tracks, results):
    # just return the frame as it is when theres not results
    if isinstance(results, dict) and results.get("status") == "warming_up":
        return frame

    results_map = {r["id"]: r for r in results} #converting it to list

    colors = {
        "Smooth Driver": (0, 200, 0), #green
        "Moderate Driver": (0, 165, 255), #orange
        "Risky Driver": (0, 0, 220), #red
    }

    for track in tracks:
        tid = track["id"]
        x1, y1, x2, y2 = track["bbox"]

        result = results_map.get(tid)

        if result:
            label = result["label"]
            score = result["score"]
            color = colors.get(label, (200, 200, 200))
        else:
            label = "Analyzing"
            score = "-"
            color = (200, 200, 200) #gray

        cv.rectangle(frame, (x1, y1), (x2,y2), color, 2)

        text = f"#{tid} | {label} | {score}"
        text_y = max(y1 - 10, 15)
        cv.putText(frame, text, (x1, text_y),cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


