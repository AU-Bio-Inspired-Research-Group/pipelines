import argparse
from calendar import c
import os
import cv2
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim
from graph import *
from methods import *
from filters import *
from crop import *
from ssim import skimageMethods
from usefulData import *
from itertools import combinations

comparison_techniques = {
        "PSNR": psnr,
        "MAE": mae,
        "NCC": ncc,
        "SSIM": lambda imgA, imgB: ssim(imgA, imgB, **kwargs),
        "Histogram": histogram_intersection,
        "EMD": emd,
        "Absolute": abs_diff,
        "Correlation": correlation,
        "Bhattacharyya": bhattacharyya,
        "Weighted": center_weighted_image_diff
    }

def pipeline_1(method, folderA, folderB, **kwargs):
    # Different because don't use openCV
    if method == "SSIM" or method == "RMSE":
        skimageMethods(method, folderA, folderB, **kwargs)
        analyze_results("skimage_results.json", method)
    else:
        iterateThroughImages(folderA, folderB, method, **kwargs)
        updated_colourmap(method, "results.json")
        analyze_results("results.json", method)

def pipeline_2(filename, folderA, folderB, outputA, outputB):
    crop_image(filename, folderA, folderB, outputA, outputB)
    pass

def resize(x_dim, y_dim, folderA, folderB):
    if x_dim is not None and y_dim is not None:
        if not (check_image_dimensions("resizedA", x_dim, y_dim) and check_image_dimensions("resizedB", x_dim, y_dim)):
            print("Resizing images to dimensions:", x_dim, y_dim)
            block_average_png_to_json(folderA, "resizedA", y_dim, x_dim)
            block_average_png_to_json(folderB, "resizedB", y_dim, x_dim)
        else:
            print("Resized images with matching dimensions already exist.")

def filterImage(filter_name, input_folderA, input_folderB, output_folderA, output_folderB, **kwargs):
    filters = {
        "greyscale": greyscale,
        "colour": replaceColour,
        "flip": flip_horizontal,
        "uniform": uniform_sub_sampling,
        "block": block_based_sub_sampling,
        "nonuniform": non_uniform_sub_sampling
        # Add other filters here as needed
    }

    if filter_name not in filters:
        raise ValueError(f"Unknown filter: {filter_name}")

    filter_function = filters[filter_name]

    # Ensure output folders exist or create them
    for folder in [output_folderA, output_folderB]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created output folder: {folder}")

    # Apply the filter to all images in both input folders and save to the respective output folders
    for folder, output_folder in [(input_folderA, output_folderA), (input_folderB, output_folderB)]:
        for filename in os.listdir(folder):
            if filename.lower().endswith('.png'):
                input_path = os.path.join(folder, filename)
                output_path = os.path.join(output_folder, filename)

                try:
                    # Apply the filter with optional parameters
                    if kwargs:
                        filter_function(input_path, output_path, **kwargs)
                    else:
                        filter_function(input_path, output_path)
                        
                    print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

def graph(method, json, bottomcoverage, topcoverage):
    updated_colourmap(method, "results.json", bottomcoverage, topcoverage)
    analyze_results("results.json", method)

def average(inputFolder, outputFile, windowSize=0):
    if windowSize == 0:
        average_images(inputFolder, outputFile)
    else:
        average_folder(inputFolder, 200, outputFile)

def find_ratio_pairs(num1, num2):
    # Calculate the original ratio
    if num2 == 0:
        print("The second number cannot be zero.")
        return
    
    original_ratio = num1 / num2

    # Iterate over possible pairs
    for i in range(1, num1):
        for j in range(1, num2):
            if i / j == original_ratio:
                print(f"({i}, {j})")

def iterateThroughImages(folderA, folderB, method, **kwargs):

    results = {}

    # Load existing results if available
    try:
        results_file = "results.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
        
            # Check if any image has data for the specified method
            if any(method in image_results for image_results in results.values()):
                user_input = input(f"Results for method '{method}' already exist. Do you want to continue processing? (yes/no): ").strip().lower()
                if user_input != 'yes':
                    print("Skipping processing.")
                    return
    except json.JSONDecodeError:
        print("Error loading existing results. Starting fresh.")

    for filename in os.listdir(folderA):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(filename)

            imageA = cv2.imread(os.path.join(folderA, filename))
            imageB = cv2.imread(os.path.join(folderB, filename))
            if imageA is None or imageB is None:
                continue

            function = comparison_techniques[method]
            try:
                result = function(imageA, imageB)
            except Exception as e:
                result = str(e)

            # Ensure the filename entry exists in results
            if filename not in results:
                results[filename] = {}

            # Update the method result for this image
            results[filename][method] = result

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f)

    print("Results saved to results.json")


def iterateThroughImagePairs(folder, method, **kwargs):
    # Define comparison techniques

    results = {}

    # Load existing results if available
    results_file = "results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Check if any image pair has data for the specified method
        if any(method in image_results for image_results in results.values()):
            user_input = input(f"Results for method '{method}' already exist. Do you want to continue processing? (yes/no): ").strip().lower()
            if user_input != 'yes':
                print("Skipping processing.")
                return

    # Get all image filenames
    image_files = [f for f in os.listdir(folder) if f.endswith(".png") or f.endswith(".jpg")]

    # Iterate through all unique pairs of images
    for file1, file2 in combinations(image_files, 2):
        print(f"Comparing {file1} and {file2}")

        imageA = cv2.imread(os.path.join(folder, file1))
        imageB = cv2.imread(os.path.join(folder, file2))
        if imageA is None or imageB is None:
            continue

        function = comparison_techniques[method]
        try:
            result = function(imageA, imageB)
        except Exception as e:
            result = str(e)

        # Ensure the filename entry exists in results
        pair_key = f"{file1} vs {file2}"
        if pair_key not in results:
            results[pair_key] = {}

        # Update the method result for this image pair
        results[pair_key][method] = result

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f)

    print("Results saved to results.json")

def check_image_dimensions(folder, x_dim, y_dim):
    # Check if folder exists
    if not os.path.exists(folder):
        return False

    # List all files in the directory
    files = os.listdir(folder)
    if not files:
        return False

    # Read the first PNG image
    first_image_path = os.path.join(folder, files[0])
    if not first_image_path.lower().endswith(".png"):
        return False

    # Get image dimensions
    image = cv2.imread(first_image_path)
    if image is None:
        return False

    # Check if dimensions match
    return image.shape[1] == x_dim and image.shape[0] == y_dim

def binaryCompare(json1, json2, method):
    plot_binary_comparison(json1, json2, method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line Interface for Pipelines")
    subparsers = parser.add_subparsers(dest="pipeline")

    # Pipeline 1
    pipeline1_parser = subparsers.add_parser("pipeline1", help="Pipeline 1")
    pipeline1_parser.add_argument("method", choices=["PSNR", "MAE", "NCC", "SSIM", "RMSE", "Histogram", "EMD", "Absolute", "Correlation", "Bhattacharyya", "Weighted"], help="Method")
    pipeline1_parser.add_argument("folderA")
    pipeline1_parser.add_argument("folderB")

    resize_parser = subparsers.add_parser("resize", help="Resize")
    resize_parser.add_argument("dimensions", nargs='*', type=int, help="Dimensions (x_dim, y_dim)")
    resize_parser.add_argument("folderA")
    resize_parser.add_argument("folderB")

    # SSIM specific arguments
    pipeline1_parser.add_argument("--gaussian_weights", type=bool, default=True, help="Use Gaussian weights for SSIM")
    pipeline1_parser.add_argument("--sigma", type=float, default=1.5, help="Sigma for SSIM")

    # Pipeline 2
    pipeline2_parser = subparsers.add_parser("pipeline2", help="Pipeline 2")
    pipeline2_parser.add_argument("filename")
    pipeline2_parser.add_argument("folderA")
    pipeline2_parser.add_argument("folderB")
    pipeline2_parser.add_argument("outputA")
    pipeline2_parser.add_argument("outputB")

    ratio_parser = subparsers.add_parser("ratio", help="Graph")
    ratio_parser.add_argument("num1", type=int)
    ratio_parser.add_argument("num2", type=int)

    graph_parser = subparsers.add_parser("graph", help="Graph")
    graph_parser.add_argument("method", help="json file")
    graph_parser.add_argument("json", help="json file")
    graph_parser.add_argument("bottomcoverage", help="coverage", type=int)
    graph_parser.add_argument("topcoverage", help="coverage", type=int)

    filter_parser = subparsers.add_parser("filter", help="Filter")
    filter_parser.add_argument("filter", choices=["greyscale", "colour", "flip", "uniform", "nonuniform", "block"], help="Filter")
    filter_parser.add_argument("input_folderA", help="Input folder A")
    filter_parser.add_argument("input_folderB", help="Input folder B")
    filter_parser.add_argument("output_folderA", help="Output folder A")
    filter_parser.add_argument("output_folderB", help="Output folder B")

    average_parser = subparsers.add_parser("average", help="Average images in a directory with a sliding window")
    average_parser.add_argument("inputFolder", type=str, help="Path to the input folder containing images")
    average_parser.add_argument("outputFile", type=str, help="Base name for the output averaged image files")
    average_parser.add_argument("--window_size", type=int, default=200, help="Number of images to average in each window (default: 200)")

    binary_parser = subparsers.add_parser("binary",)
    binary_parser.add_argument("json1", type=str)
    binary_parser.add_argument("json2", type=str)
    binary_parser.add_argument("method", type=str)


    args = parser.parse_args()

    if args.pipeline == "pipeline1":
        kwargs = {}
        if args.method == "SSIM":
            kwargs['gaussian_weights'] = args.gaussian_weights
            kwargs['sigma'] = args.sigma
        pipeline_1(args.method, args.folderA, args.folderB, **kwargs)
    elif args.pipeline == "pipeline2":
        pipeline_2(args.filename, args.folderA, args.folderB, args.outputA, args.outputB)
    elif args.pipeline == "pipeline3":
        filterImage(args.filter, args.input_folderA, args.input_folderB, args.output_folderA, args.output_folderB)
    elif args.pipeline == "graph":
        graph(args.method, args.json, args.bottomcoverage,args.topcoverage)
    elif args.pipeline == "resize":
        if args.dimensions:
            if len(args.dimensions) == 2:
                resize(args.dimensions[0], args.dimensions[1], args.folderA, args.folderB)
            else:
                print("Please provide both x_dim and y_dim.")
    elif args.pipeline == "filter":
        filterImage(args.filter, args.input_folderA, args.input_folderB, args.output_folderA, args.output_folderB)
    elif args.pipeline == "ratio":
        find_ratio_pairs(args.num1, args.num2)
    elif args.pipeline == "average":
        average(args.inputFolder, args.outputFile, args.window_size)
    elif args.pipeline == "binary":
        binaryCompare(args.json1, args.json2, args.method)

