import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
    door_positions_data = pd.read_csv("C:\\Users\\jonat\\source\\repos\\doors\\doors\\doorA_positions.csv")

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