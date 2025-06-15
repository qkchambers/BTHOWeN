import os
import numpy as np
from PIL import Image
import cv2
import json

datasets = [
            "badWeather",
            "cameraJitter",
            "intermittentObjectMotion",	
            "nightVideos",
            "shadow",
            "turbulence",
            "baseline",
            "dynamicBackground",
            "lowFramerate",
            "PTZ",
            "thermal"
            ]

videos = {
    "badWeather": ["blizzard", "skating", "snowFall", "wetSnow"],
    "baseline": ["highway", "office", "pedestrians", "PETS2006"],
    "cameraJitter": ["badminton", "boulevard", "sidewalk", "traffic"],
    "dynamicBackground": ["boats", "canoe", "fall", "fountain01", "fountain02", "overpass"],
    "intermittentObjectMotion": ["abandonedBox", "parking", "sofa", "streetLight", "tramstop", "winterDriveway"],
    "lowFramerate": ["port_0_17fps", "tramCrossroad_1fps", "tunnelExit_0_35fps", "turnpike_0_5fps"],
    "nightVideos": ["bridgeEntry", "busyBoulvard", "fluidHighway", "streetCornerAtNight", "tramStation", "winterStreet"],
    "PTZ": ["continuousPan", "intermittentPan", "twoPositionPTZCam", "zoomInZoomOut"],
    "shadow": ["backdoor", "bungalows", "busStation", "copyMachine", "cubicle", "peopleInShade"],
    "thermal": ["corridor", "diningRoom", "lakeSide", "library", "park"],
    "turbulence": ["turbulence0", "turbulence1", "turbulence2", "turbulence3"]
}

folder = "datasets/"
groundtruth = "groundtruth/"

# Read all images from groundtruth_path and convert them into a NumPy array
def load_images_to_array(groundtruth_path):
    images = []
    
    # Get a sorted list of image files in the folder
    for filename in sorted(os.listdir(groundtruth_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Filter image files
            file_path = os.path.join(groundtruth_path, filename)
            try:
                # Open the image
                img = Image.open(file_path).convert("L")  # Convert to grayscale
                images.append(np.array(img))  # Convert to NumPy array and append
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    # Stack all images into a single NumPy array
    images_array = np.stack(images)
    return images_array

# Example usage
#images_array = load_images_to_array(groundtruth_path)
#print(f"Loaded {images_array.shape[0]} images with shape {images_array.shape[1:]} each.")


def compute_metrics(gt_path, pred_path, first_index):
    TP = FP = TN = FN = 0

    for filename in sorted(os.listdir(pred_path)):
        gt_img = cv2.imread(os.path.join(gt_path, filename), cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(os.path.join(pred_path, filename), cv2.IMREAD_GRAYSCALE)

        # Apply binary threshold
        _, thresh = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)

        # Create a kernel (size affects how much erosion/dilation happens)
        kernel = np.ones((3, 3), np.uint8)

        # Apply erosion (removes noise, shrinks white areas)
        eroded = cv2.erode(thresh, kernel, iterations=1)

        # Apply dilation (expands white areas, fills gaps)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        # TODO validate this starts on the correct frame (might be off by one)
        #gt_img = gt_img[first_index:]  
        #pred_img = pred_img[first_index:] 

        # Binarize prediction to 0 or 255 if necessary
        pred_bin = (dilated > 127).astype(np.uint8) * 255

        mask = (gt_img != 170) & (gt_img != 85)  # ignore unknown and shadow
        gt_fg = (gt_img == 255)
        gt_bg = (gt_img == 0)
        pred_fg = (pred_bin == 255)
        pred_bg = (pred_bin == 0)

        TP += np.sum(pred_fg & gt_fg & mask)
        FP += np.sum(pred_fg & gt_bg & mask)
        FN += np.sum(pred_bg & gt_fg & mask)
        TN += np.sum(pred_bg & gt_bg & mask)

    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    fpr = FP / (FP + TN + 1e-8)
    fnr = FN / (FN + TP + 1e-8)
    pwc = 100 * (FN + FP) / (TP + TN + FP + FN + 1e-8)

    return {
        "Recall": recall,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "PWC": pwc,
        "Precision": precision,
        "F1 Score": f1,
    }




def run_metrics(dataset):
    first_index = get_first_index(dataset)
    groundtruth_path = "dataset/" + datasets[0] + "/" + videos['badWeather'][0] + "/groundtruth/"
    model_path = "output/" + datasets[0] + "/" + videos['badWeather'][0] + "/"
    metrics = compute_metrics(groundtruth_path, model_path, first_index)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

def find_first_labeled_frame(gt_dir):
    filenames = sorted(os.listdir(gt_dir))
    for idx, fname in enumerate(filenames):
        path = os.path.join(gt_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        unique_vals = np.unique(img)
        valid = np.setdiff1d(unique_vals, [0, 85, 170])

        if valid.size > 0:
            return idx, fname  # Found first labeled frame

    return None, None  # No labeled frames

def scan_cdnet_dataset(base_path):
    result = {}
    for category in sorted(os.listdir(base_path)):
        cat_path = os.path.join(base_path, category)
        if not os.path.isdir(cat_path):
            continue

        for sequence in sorted(os.listdir(cat_path)):
            seq_path = os.path.join(cat_path, sequence)
            gt_path = os.path.join(seq_path, "groundtruth")

            if not os.path.exists(gt_path):
                continue

            idx, fname = find_first_labeled_frame(gt_path)
            result[f"{category}/{sequence}"] = {
                "index": idx,
                "filename": fname
            }

    return result



def save_groundtruth_first_index():
    # Example usage
    base_dataset_path = "dataset"  # Change to your actual dataset root
    first_labeled_frames = scan_cdnet_dataset(base_dataset_path)

    # Save results
    output_file = "first_labeled_frames.json"
    with open(output_file, "w") as f:
        json.dump(first_labeled_frames, f, indent=2)

    print(f"Saved first labeled frame info to {output_file}")

# file comes in as baseline/highway/, so I remove last '/'
def get_first_index(filename):
    # Load the JSON file
    with open("first_labeled_frames.json", "r") as f:
        data = json.load(f)

    # Extract the index and filename for each sequence

    return data[filename[:-1]]["index"]


def run_metrics1(dataset):
    first_index = get_first_index(dataset)
    groundtruth_path = f"dataset/{dataset}groundtruth/"
    model_path = f"output/{dataset}"
    metrics = compute_metrics(groundtruth_path, model_path, first_index)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

run_metrics1('lowFramerate/turnpike_0_5fps/')

#print(get_first_index('baseline/highway'))
