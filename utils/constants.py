# 2D DATASET
COLORS = ["blue", "green", "grey", "purple", "red", "yellow"]
OBJECT_TYPES = ["ball", "box", "key"]
DIRECTIONS = ["down", "to the left", "to the right", "up"]


# 3D DATASET
BLOCK_COLORS = ["blue", "green", "grey", "purple", "red", "yellow"]
BOWL_COLORS = ["brown", "cyan", "orange", "petrol", "pink", "white"]
SIZES = ["big", "small"]

CONCEPT_TO_IDX = {
    "2d": {"color": 0, "type": 1, "direction": 2},
    "3d": {"block": 0, "bowl": 1},
}

INIT_TASKS = {
    "2d": [
        ("blue", "ball", "to the right"),
        ("blue", "box", "up"),
        ("green", "box", "to the left"),
        ("green", "key", "up"),
        ("grey", "box", "down"),
        ("grey", "key", "to the left"),
        ("purple", "ball", "to the left"),
        ("purple", "key", "to the right"),
        ("red", "ball", "up"),
        ("red", "key", "down"),
        ("yellow", "ball", "down"),
        ("yellow", "box", "to the right"),
    ],
    "3d": [
        ("blue", "big", "pink"),
        ("blue", "small", "brown"),
        ("green", "big", "brown"),
        ("green", "small", "petrol"),
        ("grey", "big", "cyan"),
        ("grey", "small", "white"),
        ("purple", "big", "petrol"),
        ("purple", "small", "pink"),
        ("red", "big", "orange"),
        ("red", "small", "cyan"),
        ("yellow", "big", "white"),
        ("yellow", "small", "orange"),
    ],
}
