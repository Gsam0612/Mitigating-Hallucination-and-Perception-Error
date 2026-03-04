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

# Maps common VLM words/synonyms → canonical COCO category name.
# This lets us recognise "man" → "person", "sofa" → "couch", etc.
SYNONYM_TO_CATEGORY = {
    # person synonyms
    "man": "person", "woman": "person", "men": "person", "women": "person",
    "boy": "person", "girl": "person", "child": "person", "children": "person",
    "kid": "person", "kids": "person", "people": "person", "player": "person",
    "rider": "person", "pedestrian": "person", "skier": "person",
    "surfer": "person", "biker": "person", "baby": "person",
    # vehicle synonyms
    "bike": "bicycle", "cycle": "bicycle", "motorbike": "motorcycle",
    "plane": "airplane", "jet": "airplane", "vehicle": "car",
    "automobile": "car", "van": "truck", "lorry": "truck",
    "ship": "boat", "sailboat": "boat", "yacht": "boat",
    # furniture / household
    "sofa": "couch", "loveseat": "couch", "settee": "couch",
    "television": "tv", "tv screen": "tv", "telly": "tv",
    "telephone": "cell phone", "phone": "cell phone", "mobile": "cell phone",
    "fridge": "refrigerator", "stool": "chair", "armchair": "chair",
    "desk": "dining table", "nightstand": "bed",
    "plant": "potted plant", "houseplant": "potted plant", "flower": "potted plant",
    # food / kitchen
    "doughnut": "donut", "hotdog": "hot dog",
    "wineglass": "wine glass", "goblet": "wine glass",
    # animal synonyms
    "kitten": "cat", "puppy": "dog", "pup": "dog",
    "calf": "cow", "cattle": "cow", "bull": "cow",
    "lamb": "sheep", "ram": "sheep",
    # accessories
    "bag": "handbag", "purse": "handbag", "luggage": "suitcase",
    "necktie": "tie", "bow tie": "tie",
    "rucksack": "backpack", "knapsack": "backpack",
    # sport
    "ball": "sports ball", "ski": "skis",
    "bat": "baseball bat", "glove": "baseball glove",
    # plural forms of common categories
    "cars": "car", "trucks": "truck", "buses": "bus",
    "chairs": "chair", "books": "book", "bottles": "bottle",
    "cups": "cup", "bowls": "bowl", "birds": "bird",
    "cats": "cat", "dogs": "dog", "horses": "horse",
    "cows": "cow", "elephants": "elephant", "bears": "bear",
    "zebras": "zebra", "giraffes": "giraffe", "sheep": "sheep",
    "persons": "person", "benches": "bench", "boats": "boat",
    "trains": "train", "bicycles": "bicycle", "motorcycles": "motorcycle",
    "airplanes": "airplane", "clocks": "clock", "vases": "vase",
    "couches": "couch", "beds": "bed", "toilets": "toilet",
    "laptops": "laptop", "keyboards": "keyboard", "ovens": "oven",
    "knives": "knife", "forks": "fork", "spoons": "spoon",
    "bananas": "banana", "apples": "apple", "oranges": "orange",
    "pizzas": "pizza", "donuts": "donut", "cakes": "cake",
    "sandwiches": "sandwich", "carrots": "carrot",
    "umbrellas": "umbrella", "kites": "kite",
    "skateboards": "skateboard", "surfboards": "surfboard",
    "suitcases": "suitcase", "backpacks": "backpack",
}
