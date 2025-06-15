#!/usr/bin/env python3

import sys
import pickle
import lzma
import argparse
import numpy as np
from PIL import Image
from train_swept_models import get_datasets, binarize_datasets, run_inference, get_dataset_images, get_first_index, convert_to_grayscale, run_inference_image

def eval_model(model_fname, dset_name):
    print("Loading model")
    with lzma.open(model_fname, "rb") as f:
        state_dict = pickle.load(f)
    if not hasattr(state_dict["model"], "pad_zeros"):
        state_dict["model"].pad_zeros = 0

    print("Loading dataset")
    train_dataset, test_dataset = get_datasets(dset_name)

    print("Running inference")
    bits_per_input = state_dict["info"]["bits_per_input"]
    test_inputs, test_labels = binarize_datasets(train_dataset, test_dataset, bits_per_input)[-2:]
    result = run_inference(test_inputs, test_labels, state_dict["model"], 1)

def read_arguments():
    parser = argparse.ArgumentParser(description="Run inference on a pre-trained BTHOWeN model")
    parser.add_argument("model_fname", help="Path to pretrained model .pickle.lzma")
    parser.add_argument("dset_name", help="Name of dataset to use for inference; obviously this must match the model")
    args = parser.parse_args()
    return args

def test_model(model_fname, dset_name):
    print("Loading model")
    with lzma.open(model_fname, "rb") as f:
        state_dict = pickle.load(f)
    if not hasattr(state_dict["model"], "pad_zeros"):
        state_dict["model"].pad_zeros = 0

    print("Loading dataset")
    input_images = get_dataset_images(dset_name+"input/")

    # TODO: Might be good to very I am starting at the right index
    first_index = get_first_index(dset_name)
    gray_images = convert_to_grayscale(input_images)

    np_images = [np.array(img) for img in gray_images]
    gray_array = np.stack(np_images)
    # TODO is width and heigh correct?
    number_of_images, width, height = gray_array.shape
    flattened_gray_array = gray_array.flatten()
    pixel_position_tuples = [([pixel_value], index % (width*height)) for index, pixel_value in enumerate(flattened_gray_array)]
    

    train_dataset = pixel_position_tuples[:(first_index-1)*width*height]
    test_dataset = pixel_position_tuples[(first_index*width*height):]
    bits_per_input = state_dict["info"]["bits_per_input"]
    datasets = binarize_datasets(train_dataset, test_dataset, bits_per_input)
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = datasets

    print("Running inference")

    folder_name = f"output/{dset_name}/"
    number_of_test_images = int(len(test_inputs)/(width*height))
    for i in range(number_of_test_images):
        motion_image = run_inference_image(test_inputs[i*width*height:(i+1)*width*height], test_labels[i*width*height:(i+1)*width*height], state_dict["model"], bleach=467)
        array_scaled = (np.array(motion_image) * 255).astype(np.uint8)
        array_scaled = array_scaled.reshape((width, height)) 
        bw_image = Image.fromarray(array_scaled, mode="L") 
        file_name = f"gt{first_index+i+1:06d}"
        bw_image.save(folder_name+file_name+".png")


if __name__ == "__main__":
    args = read_arguments()
    #eval_model(args.model_fname, args.dset_name)
    test_model(args.model_fname, args.dset_name)

