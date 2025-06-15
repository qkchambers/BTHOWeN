#!/usr/bin/env python3

import sys
import itertools
import argparse
import ctypes as c
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
from scipy.stats import norm

from PIL import Image
# For saving models
import pickle
import lzma
import os

from wisard import WiSARD
from benchmarks import get_first_index

# For the tabular datasets (all except MNIST)
import tabular_tools

# Perform inference operations using provided test set on provided model with specified bleaching value (default 1)
def run_inference(inputs, labels, model, bleach=1):
    num_samples = len(inputs)
    correct = 0
    ties = 0
    model.set_bleaching(bleach)
    for d in range(num_samples):
        prediction = model.predict(inputs[d])
        label = labels[d]
        if len(prediction) > 1:
            ties += 1
        if prediction[0] == label:
            correct += 1
    correct_percent = round((100 * correct) / num_samples, 4)
    tie_percent = round((100 * ties) / num_samples, 4)
    print(f"With bleaching={bleach}, accuracy={correct}/{num_samples} ({correct_percent}%); ties={ties}/{num_samples} ({tie_percent}%)")
    return correct

def run_inference_image(inputs, labels, model, bleach=1):
    image = []
    num_samples = len(inputs)
    model.set_bleaching(bleach)
    for d in range(num_samples):
        prediction = model.predict(inputs[d], d)

        image.append(1)
        if prediction >= 1:
            image[d] = 0

    return image

def parameterized_run(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, unit_inputs, unit_entries, unit_hashes):
    model = WiSARD(train_inputs[0].size, train_labels.max()+1, unit_inputs, unit_entries, unit_hashes)

    print("Training model")
    for d in range(len(train_inputs)):
        model.train(train_inputs[d], train_labels[d])
        if ((d+1) % 10000) == 0:
            print(d+1)

    max_val = 0
    for d in model.discriminators:
        for f in d.filters:
            max_val = max(max_val, f.data.max())
    print(f"Maximum possible bleach value is {max_val}")
    # Use a binary search-based strategy to find the value of b that maximizes accuracy on the validation set
    best_bleach = max_val // 2
    step = max(max_val // 4, 1)
    bleach_accuracies = {}
    while True:
        values = [best_bleach-step, best_bleach, best_bleach+step]
        accuracies = []
        for b in values:
            if b in bleach_accuracies:
                accuracies.append(bleach_accuracies[b])
            elif b < 1:
                accuracies.append(0)
            else:
                accuracy = run_inference(val_inputs, val_labels, model, b)
                bleach_accuracies[b] = accuracy
                accuracies.append(accuracy)
        new_best_bleach = values[accuracies.index(max(accuracies))]
        if (new_best_bleach == best_bleach) and (step == 1):
            break
        best_bleach = new_best_bleach
        if step > 1:
            step //= 2
    print(f"Best bleach: {best_bleach}; inputs/entries/hashes = {unit_inputs},{unit_entries},{unit_hashes}")
    # Evaluate on test set
    print("Testing model")
    accuracy = run_inference(test_inputs, test_labels, model, bleach=best_bleach)
    return model, accuracy

# Convert input dataset to binary representation
# Use a thermometer encoding with a configurable number of bits per input
# A thermometer encoding is a binary encoding in which subsequent bits are set as the value increases
#  e.g. 0000 => 0001 => 0011 => 0111 => 1111
def binarize_datasets(train_dataset, test_dataset, bits_per_input, separate_validation_dset=False, train_val_split_ratio=0.9):
    # Given a Gaussian with mean=0 and std=1, choose values which divide the distribution into regions of equal probability
    # This will be used to determine thresholds for the thermometer encoding
    std_skews = [norm.ppf((i+1)/(bits_per_input+1))
                 for i in range(bits_per_input)]

    """
    probs = [(i+1)/(bits_per_input+1) for i in range(bits_per_input)]
    mirrored_probs = [abs(p - 0.5) * 2 for p in probs]  # 0 at center, 1 at edges
    std_skews = [norm.ppf(p/2) if p < 0.5 else norm.ppf(1 - (p/2)) for p in probs]
    """

    print("Binarizing train/validation dataset")
    train_inputs = []
    train_labels = []
    train_inputs = train_dataset
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)

    use_gaussian_encoding = False
    if use_gaussian_encoding:
        mean_inputs = train_inputs.mean(axis=0)
        std_inputs = train_inputs.std(axis=0)
        train_binarizations = []
        for i in std_skews:
            train_binarizations.append(
                (train_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        min_inputs = train_inputs.min(axis=0)
        max_inputs = train_inputs.max(axis=0)
        train_binarizations = []
        for i in range(bits_per_input):
            train_binarizations.append(
                (train_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))

    # Creates thermometer encoding
    train_inputs = np.concatenate(train_binarizations, axis=1)

    # Ideally, we would perform bleaching using a separate dataset from the training set
    #  (we call this the "validation set", though this is arguably a misnomer),
    #  since much of the point of bleaching is to improve generalization to new data.
    # However, some of the datasets we use are simply too small for this to be effective;
    #  a very small bleaching/validation set essentially fits to random noise,
    #  and making the set larger decreases the size of the training set too much.
    # In these cases, we use the same dataset for training and validation
    if separate_validation_dset is None:
        separate_validation_dset = (len(train_inputs) > 10000)
    if separate_validation_dset:
        split = int(train_val_split_ratio*len(train_inputs))
        val_inputs = train_inputs[split:]
        val_labels = train_labels[split:]
        train_inputs = train_inputs[:split]
        train_labels = train_labels[:split]
    else:
        val_inputs = train_inputs
        val_labels = train_labels

    print("Binarizing test dataset")
    test_inputs = []
    test_labels = []
    for d in test_dataset:
        # Expects inputs to be already flattened numpy arrays
        test_inputs.append(d[0])
        #test_labels.append(d[1])
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)
    test_binarizations = []
    #test_inputs = np.concatenate(test_binarizations, axis=1)

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels

def get_datasets(dset_name):
    dset_name = dset_name.lower()
    print(f"Loading dataset ({dset_name})")
    if dset_name == 'mnist':
        train_dataset = dsets.MNIST(
            root='./data',
            train=True,
            transform=transforms.ToTensor(),
            download=True)
        new_train_dataset = []
        for d in train_dataset:
            new_train_dataset.append((d[0].numpy().flatten(), d[1]))
        train_dataset = new_train_dataset
        test_dataset = dsets.MNIST(
            root='./data',
            train=False,
            transform=transforms.ToTensor())
        new_test_dataset = []
        for d in test_dataset:
            new_test_dataset.append((d[0].numpy().flatten(), d[1]))
        test_dataset = new_test_dataset
    else:
        train_dataset, test_dataset = tabular_tools.get_dataset(dset_name)
    return train_dataset, test_dataset

def create_models(dset_name, unit_inputs, unit_entries, unit_hashes, bits_per_input, num_workers, save_prefix="model"):
    train_dataset, test_dataset = get_datasets(dset_name)

    datasets = binarize_datasets(train_dataset, test_dataset, bits_per_input)
    prod = list(itertools.product(unit_inputs, unit_entries, unit_hashes))
    configurations = [datasets + c for c in prod]

    if num_workers == -1:
        num_workers = cpu_count()
    print(f"Launching jobs for {len(configurations)} configurations across {num_workers} workers")
    with Pool(num_workers) as p:
        results = p.starmap(parameterized_run, configurations)
    for entries in unit_entries:
        print(
            f"Best with {entries} entries: {max([results[i][1] for i in range(len(results)) if configurations[i][7] == entries])}")
    configs_plus_results = [[configurations[i][6:9]] +
                            list(results[i]) for i in range(len(results))]
    configs_plus_results.sort(reverse=True, key=lambda x: x[2])
    for i in configs_plus_results:
        print(f"{i[0]}: {i[2]} ({i[2] / len(datasets[4])})")

    # Ensure folder for dataset exists
    os.makedirs(os.path.dirname(f"./models/{dset_name}/{save_prefix}"), exist_ok=True)

    for idx, result in enumerate(results):
        model = result[0]
        model_inputs, model_entries, model_hashes = configurations[idx][6:9]
        save_model(model, (datasets[0][0].size // bits_per_input),
            f"./models/{dset_name}/{save_prefix}_{model_inputs}input_{model_entries}entry_{model_hashes}hash_{bits_per_input}bpi.pickle.lzma")

def save_model(model, num_inputs, fname):
    model.binarize()
    # TODO bits_per_input is caluclated wrong because I pass a wrong value for num_inputs
    model_info = {
        "num_inputs": num_inputs,
        "num_classes": len(model.discriminators),
        "bits_per_input": len(model.input_order) // num_inputs,
        "num_filter_inputs": model.discriminators[0].filters[0].num_inputs,
        "num_filter_entries": model.discriminators[0].filters[0].num_entries,
        "num_filter_hashes": model.discriminators[0].filters[0].num_hashes,\
        "hash_values": model.discriminators[0].filters[0].hash_values
    }
    state_dict = {
        "info": model_info,
        "model": model
    }

    with lzma.open(fname, "wb") as f:
        pickle.dump(state_dict, f)

def read_arguments():
    parser = argparse.ArgumentParser(description="Train BTHOWeN models for a dataset with specified hyperparameter sweep")
    parser.add_argument("dset_name", help="Name of dataset to use")
    parser.add_argument("--filter_inputs", nargs="+", required=True, type=int,\
            help="Number of inputs to each Bloom filter (accepts multiple values)")
    parser.add_argument("--filter_entries", nargs="+", required=True, type=int,\
            help="Number of entries in each Bloom filter (accepts multiple values; must be powers of 2)")
    parser.add_argument("--filter_hashes", nargs="+", required=True, type=int,\
            help="Number of distinct hash functions for each Bloom filter (accepts multiple values)")
    parser.add_argument("--bits_per_input", nargs="+", required=True, type=int,\
            help="Number of thermometer encoding bits for each input in the dataset (accepts multiple values)")
    parser.add_argument("--save_prefix", default="model", help="Partial path/fname to prepend to each output file")
    parser.add_argument("--num_workers", default=-1, type=int, help="Number of processes to run in parallel; defaults to number of logical CPUs")
    args = parser.parse_args()
    return args

def get_dataset_images(file_path):
    # Example usage
    folder_path = "dataset/"  # Replace with your folder path
    #file_path = ""
    images = read_images_from_folder(folder_path+file_path)

    # Print the number of images loaded
    print(f"Loaded {len(images)} images.")
    return images
    


def read_images_from_folder(folder_path):
    
    images = []
    
    # Get a sorted list of files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Filter image files
            file_path = os.path.join(folder_path, filename)
            try:
                # Open the image and append it to the list
                #print(f"Loading image: {filename}")
                img = Image.open(file_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    return images

def convert_to_grayscale(images):
    gray_images = []
    for image in images:
        #image_array = np.array(image)
        #print(image_array)
        gray_images.append(image.convert("L"))  # Convert to grayscale

    return gray_images


def change_detection_2(dset_name, unit_inputs, unit_entries, unit_hashes, bits_per_input, num_workers, save_prefix="model"):
  
    # Configure dataset
    input_images = np.stack([
        np.array(img.convert("L")) for img in get_dataset_images(dset_name + "input/")
    ])  # Load and convert to grayscale in one step

    # Get the starting index
    first_index = get_first_index(dset_name) + 200 # TODO put back to normal

    # Get dimensions
    number_of_images, width, height = input_images.shape

    # Split into train and test datasets
    input_images = input_images.reshape(number_of_images, -1).reshape(-1, 1)
    train_dataset = input_images #pixel_position_tuples #[:first_index * width * height]
    test_dataset = input_images[:width*height]#pixel_position_tuples[:width * height]

    # Binarize datasets
    datasets = binarize_datasets(train_dataset, test_dataset, bits_per_input)
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = datasets


    # TODO need to remove the labels
    model = WiSARD(train_inputs[0].size, width*height+1, unit_inputs[0], unit_entries[0], unit_hashes[0])

    # Train model
    print("Training model")
    # Do initial training here
    """
    initial_train_length = 50 * width * height
    for d in range(initial_train_length):
        model.train(train_inputs[d], d % (width*height))
        if ((d+1) % 10000) == 0:
            print(d+1)

    max_val = 0
    for d in model.discriminators:
        for f in d.filters:
            max_val = max(max_val, f.data.max())
    print(f"Maximum possible bleach value is {max_val}")
    """

    # Save model
    # This causes chaos
    """
    num_inputs = 1
    os.makedirs(os.path.dirname(f"{save_prefix}/{dset_name}"), exist_ok=True)
    fname = f"{save_prefix}/{dset_name}/filterinputs_{unit_inputs[0]}input_{unit_entries[0]}entry_{unit_hashes[0]}hash_{bits_per_input}bpi.pickle.lzma"
    save_model(model, num_inputs, fname)
    """
    

    # Test model
    buffer_size = 20
    buffer_start = 10 * width * height
    buffer_end = buffer_size * width * height + buffer_start
    
    bleach = 5
    buffer_count = 0
    folder_name = f"output/{dset_name}/"
    os.makedirs(os.path.dirname(folder_name), exist_ok=True)
    number_of_test_images = int((len(train_inputs)/(width*height))-first_index-buffer_size)
    for i in range(number_of_test_images):
        # Need a history buffer (maybe just use train set becuase of this)
        start = (first_index-buffer_size+i)*width*height
        end = (first_index-buffer_size+i+1)*width*height
        if(start > first_index*width*height):
            motion_image = run_inference_image(train_inputs[start:end], [], model, bleach=bleach)
            array_scaled = (np.array(motion_image) * 255).astype(np.uint8)
            array_scaled = array_scaled.reshape((width, height)) 
            bw_image = Image.fromarray(array_scaled, mode="L") 
            file_name = f"gt{first_index-buffer_size+i+1:06d}"
            print(f"Saving image: {folder_name+file_name}.png")
            bw_image.save(folder_name+file_name+".png")

        # History buffer
        buffer_count += 1
        for j in range(width*height):
            model.train(train_inputs[(start-buffer_start+j)], j)
            if buffer_count > buffer_size:
                model.remove(train_inputs[(start-buffer_end+j)], j)


def change_detection(dset_name, unit_inputs, unit_entries, unit_hashes, bits_per_input, num_workers, save_prefix="model"):
  
    buffer_size = 100
    # Configure dataset
    input_images = np.stack([
        np.array(img) for img in get_dataset_images(dset_name + "input/")
    ]) 
    first_index = get_first_index(dset_name) - 1 #+ 200

    #input_images = input_images[first_index-buffer_size-1:]
    # Get dimensions
    # width and height are swapped here but doesn't matter too much right now
    number_of_images, width, height, colors = input_images.shape

    # Split into train and test datasets
    input_images = input_images.reshape(number_of_images*width *height, 3)
    train_dataset = input_images #pixel_position_tuples #[:first_index * width * height]
    test_dataset = input_images[:width*height]#pixel_position_tuples[:width * height]

    # Binarize datasets
    datasets = binarize_datasets(train_dataset, test_dataset, bits_per_input)
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = datasets


    # TODO Start from before first_index but only create image after first_index
    # Maybe remove the pretraining too
    model = WiSARD(train_inputs[0].size, width*height+1, unit_inputs[0], unit_entries[0], unit_hashes[0])

    # Train model
    print("Training model")
    # Do initial training here
    
    """
    initial_train_length = 30 * width * height
    for d in range(initial_train_length):
        model.train(train_inputs[d], d % (width*height))
        if ((d+1) % 10000) == 0:
            print(d+1)
    """

    """
    max_val = 0
    for d in model.discriminators:
        for f in d.filters:
            max_val = max(max_val, f.data.max())
    print(f"Maximum possible bleach value is {max_val}")
    """

    # Save model
    # This causes chaos
    """
    num_inputs = 1
    os.makedirs(os.path.dirname(f"{save_prefix}/{dset_name}"), exist_ok=True)
    fname = f"{save_prefix}/{dset_name}/filterinputs_{unit_inputs[0]}input_{unit_entries[0]}entry_{unit_hashes[0]}hash_{bits_per_input}bpi.pickle.lzma"
    save_model(model, num_inputs, fname)
    """
    

    # Test model
    # Test model
    
    buffer_start = 0 * width * height
    number_pixels = width * height
    
    bleach = 10
    buffer_count = 0
    folder_name = f"output/{dset_name}/"
    os.makedirs(os.path.dirname(folder_name), exist_ok=True)

    for i in range(first_index-buffer_size, number_of_images+1):
        # Need a history buffer (maybe just use train set becuase of this)
        start = i
        end = i+1
        if(start >= first_index):
            motion_image = run_inference_image(train_inputs[(start-1)*number_pixels:(end-1)*number_pixels], [], model, bleach=bleach)
            array_scaled = (np.array(motion_image) * 255).astype(np.uint8)
            array_scaled = array_scaled.reshape((width, height)) 
            bw_image = Image.fromarray(array_scaled, mode="L") 
            file_name = f"gt{start:06d}"
            print(f"Saving image: {folder_name+file_name}.png")
            bw_image.save(folder_name+file_name+".png")

        # History buffer
        buffer_count += 1
        print(f"adding {start}")
        if buffer_count > buffer_size:
            print(f"removing {start-buffer_size}")
        for j in range(number_pixels):
            model.train(train_inputs[((start-1)*number_pixels)+j], j)
            
            if buffer_count > buffer_size:
                model.remove(train_inputs[(start-1-buffer_size)*number_pixels+j], j)
                pass
 
           

def main():
    args = read_arguments()

    for bpi in args.bits_per_input:
        print(f"Do runs with {bpi} bit(s) per input")
        
        
        change_detection(args.dset_name, args.filter_inputs, args.filter_entries, args.filter_hashes,
            bpi, args.num_workers, args.save_prefix)
        """
        create_models(
            args.dset_name, args.filter_inputs, args.filter_entries, args.filter_hashes,
            bpi, args.num_workers, args.save_prefix)
        """

if __name__ == "__main__":
    main()

