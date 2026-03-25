import csv

from logger import logger


def generate_report(latest_results, filename="../report.csv"):
    if not latest_results:
        logger.warning("No results to report")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        #header row
        writer.writerow(["vehicle_id", "label", "score", "speeding", "aggressive_acc", "erratic"])

        #one row per vehicle
        for vid, result in latest_results.items():
            flags = result.get("flags", {})
            writer.writerow([
                vid,
                result["label"],
                result["score"],
                flags.get("speeding", False),
                flags.get("aggressive_acc", False),
                flags.get("erratic", False)
            ])

    logger.info(f"Report saved to {filename}")
