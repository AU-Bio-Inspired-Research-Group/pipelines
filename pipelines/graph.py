import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def colourmap(method, results_file):
    # Read image comparisons data
    with open(results_file, "r") as f:
        image_comparisons_data = json.load(f)

    # Read door positions data
    door_positions_data = pd.read_csv("doorA_positions.csv")

    # Extract the specified metric values or use direct values if method is None
    metric_values = {}
    if method is None:
        metric_values = image_comparisons_data
    else:
        for image_name, metrics in image_comparisons_data.items():
            metric_values[image_name] = metrics[method]

    # Merge metric values with door positions
    door_positions_data[method if method else "metric"] = door_positions_data["image"].map(metric_values)

    # Print original min and max values before normalization
    metric_column = method if method else "metric"
    original_min_metric = door_positions_data[metric_column].min()
    original_max_metric = door_positions_data[metric_column].max()
    print(f"Original min {metric_column}:", original_min_metric)
    print(f"Original max {metric_column}:", original_max_metric)

    # Calculate the normalization factors
    metric_range = original_max_metric - original_min_metric

    # Normalize the metric values to range between 0 and 1
    door_positions_data[f"{metric_column}_normalized"] = door_positions_data[metric_column] / original_max_metric

    # Calculate the minimum color scale value for the color maps
    min_metric_color_value = original_min_metric / original_max_metric

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot normalized metric (0 to 1)
    normalized_map = axs[0].scatter(door_positions_data["x"], door_positions_data["y"], c=door_positions_data[f"{metric_column}_normalized"], cmap='rainbow', norm=Normalize(vmin=0, vmax=1))
    axs[0].set_title(f'{metric_column} (Normalized 0 to 1)')
    axs[0].set_axis_off()  # Remove x and y axes
    fig.colorbar(normalized_map, ax=axs[0], label=metric_column)

    # Plot normalized metric (relative to minimum)
    relative_norm = Normalize(vmin=min_metric_color_value, vmax=1)
    relative_map = axs[1].scatter(door_positions_data["x"], door_positions_data["y"], c=door_positions_data[f"{metric_column}_normalized"], cmap='rainbow', norm=relative_norm)
    axs[1].set_title(f'{metric_column} (Relative to Min)')
    axs[1].set_axis_off()  # Remove x and y axes
    fig.colorbar(relative_map, ax=axs[1], label=metric_column)

    plt.tight_layout()
    plt.show()


def updated_colourmap(method, results_file, bottomCoverage=0, topCoverage=0):
    # Read image comparisons data
    with open(results_file, "r") as f:
        image_comparisons_data = json.load(f)

    # Read door positions data
    door_positions_data = pd.read_csv("360.csv")

    # Extract the specified metric values or use direct values if method is None
    metric_values = {}
    if method is None:
        metric_values = image_comparisons_data
    else:
        for image_name, metrics in image_comparisons_data.items():
            metric_values[image_name] = metrics[method]

    # Merge metric values with door positions
    metric_column = method if method else "metric"
    door_positions_data[metric_column] = door_positions_data["image"].map(metric_values)

    # Determine the thresholds based on the bottom and top coverage
    if bottomCoverage > 0:
        bottom_threshold = door_positions_data[metric_column].quantile(bottomCoverage / 100)
    else:
        bottom_threshold = door_positions_data[metric_column].min()

    if topCoverage > 0:
        top_threshold = door_positions_data[metric_column].quantile(1 - topCoverage / 100)
    else:
        top_threshold = door_positions_data[metric_column].max()

    # Filter the data based on the thresholds
    filtered_data = door_positions_data[
        (door_positions_data[metric_column] >= bottom_threshold) &
        (door_positions_data[metric_column] <= top_threshold)
    ]

    # Print original min and max values
    original_min_metric = filtered_data[metric_column].min()
    original_max_metric = filtered_data[metric_column].max()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_data["x"], filtered_data["y"], c=filtered_data[metric_column], cmap='rainbow', norm=Normalize(vmin=original_min_metric, vmax=original_max_metric))
    plt.title(f'{metric_column}')
    plt.axis('off')  # Remove x and y axes
    plt.colorbar(label=metric_column)
    plt.show()

def plot_binary_comparison(json_file1, json_file2, method):
    # Load JSON data from both files
    with open(json_file1, 'r') as f:
        data1 = json.load(f)
    
    with open(json_file2, 'r') as f:
        data2 = json.load(f)
    
    # Read door positions data from CSV
    door_positions_data = pd.read_csv("doorA_positions.csv")
    
    # Extract image names from JSON files (assuming they are identical in both files)
    image_names = list(data1.keys())  # Using data1 keys assuming they are the same for both
    
    # Initialize a numpy array to store binary comparison results
    comparison_result = np.zeros(len(image_names), dtype=int)
    
    # Perform comparison and print results for each image
    for i, img in enumerate(image_names):
        corr_value_file1 = data1[img]["Correlation"]
        corr_value_file2 = data2[img]["Correlation"]
        
        # Determine which file has the larger correlation value
        if corr_value_file1 > corr_value_file2:
            comparison_result[i] = 1  # File 1 has larger correlation value
            larger_file = json_file1
        else:
            comparison_result[i] = 0  # File 2 has larger or equal correlation value
            larger_file = json_file2
        
        # Print filename and correlation values
        print(f"Image: {img}")
        print(f"  {json_file1}: Correlation = {corr_value_file1}")
        print(f"  {json_file2}: Correlation = {corr_value_file2}")
        print(f"  Larger file: {larger_file}")
    
    # Plotting the binary comparison result using x, y positions from CSV
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')  # Choose a perceptually uniform colormap
    plt.scatter(door_positions_data["x"], door_positions_data["y"], c=comparison_result, cmap=cmap)
    plt.title(f'Binary Comparison Result for Correlation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.colorbar(label='Comparison Result (1: File 1 > File 2, 0: File 2 >= File 1)')
    plt.show()