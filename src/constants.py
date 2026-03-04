"""
Shared constants used across the project.

Single source of truth for COCO category names to ensure
consistent object matching between reward, metrics, and verification.
"""

# Standard COCO 80 categories + common visual synonyms
COCO_CATEGORIES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
    # Common synonyms and additional terms found in COCO captions
    "table", "plate", "glass", "lamp", "mug", "monitor", "screen",
    "counter", "shelf", "cabinet", "stove", "countertop", "pan", "pot",
    "pillow", "blanket", "curtain", "door", "window",
}
