import logging

logger = logging.getLogger("driver_analysis")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

handler.setFormatter(formatter)
logger.addHandler(handler)
