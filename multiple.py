import subprocess

# List of datasets
datasets = [
    #"shadow/bungalows/",
    #"shadow/peopleInShade/",
    #"baseline/pedestrians",
    #"baseline/PETS2006/",   #TODO
    #"cameraJitter/badminton/", #TODO
    #"cameraJitter/sidewalk/",
    #"cameraJitter/traffic/",
    #"dynamicBackground/canoe/",
    #"dynamicBackground/fountain01/",
    #"dynamicBackground/fountain02/",
    #"lowFramerate/tramCrossroad_1fps/",
    #"lowFramerate/turnpike_0_5fps/",
    #"nightVideos/fluidHighway/",
    #"nightVideos/winterStreet/",
    #"PTZ/continuousPan/",
    "thermal/park/",
]

# Path to train_swept_models.py
script_path = "./software_model/train_swept_models.py"

# Common arguments
common_args = [
    "--filter_inputs", "45",
    "--filter_entries", "32",
    "--filter_hashes", "4",
    "--bits_per_input", "60",
    "--save_prefix", "model",
]

# Iterate over datasets and call train_swept_models.py
for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    args = ["python3", script_path, dataset] + common_args
    subprocess.run(args, check=True)