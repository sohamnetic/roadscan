"""The class list for our own labelled frames, shared by make_tasks.py and to_yolo.py.

The order is the YOLO class index. Damage classes come first. The look-alikes after them are
labelled so the model learns that they are NOT potholes; fusion.py treats them as negatives.
"""
CLASSES = ["pothole", "alligator_crack", "crack", "speed_breaker", "manhole", "patch", "puddle"]

# How model outputs (RDD2022-trained or older names) map onto the labelling classes.
ALIASES = {
    "D00": "crack",
    "D10": "crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    "longitudinal_crack": "crack",
    "transverse_crack": "crack",
}


def canonical(name: str) -> str | None:
    name = ALIASES.get(name, name)
    return name if name in CLASSES else None
