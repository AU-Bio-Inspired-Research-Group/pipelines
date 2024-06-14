# Pipelines

To use pipelines, clone the repository and `cd` into the pipelines directory. Use `python pipelines.py [METHOD]` e.g., `python pipelines.py pipeline1 MAE doorA doorB outA outB`.

## Pipeline 1
Command: `pipeline1`

### Description
This command executes Pipeline 1 operations.

### Arguments
- `method`: Method to use for comparison.
  - Choices: ["PSNR", "MAE", "NCC", "SSIM", "Histogram", "EMD", "Absolute", "Correlation", "Bhattacharyya"]
- `folderA`: Path to folder A containing images.
- `folderB`: Path to folder B containing images.

### SSIM Specific Arguments
- `--gaussian_weights`: Use Gaussian weights for SSIM.
  - Type: bool, default: True
- `--sigma`: Sigma value for SSIM.
  - Type: float, default: 1.5

## Pipeline 2
Command: `pipeline2`

### Description
This command executes Pipeline 2 operations.

### Arguments
- `filename`: Filename for pipeline operations.
- `folderA`: Path to folder A containing images.
- `folderB`: Path to folder B containing images.
- `outputA`: Output folder A for saving results.
- `outputB`: Output folder B for saving results.

## Pipeline 3
Command: `pipeline3`

### Description
This command applies a filter to images in two input folders and saves the results in two output folders.

### Arguments
- `filter`: Filter to apply to images.
  - Choices: ["greyscale", "colour", "flip"]
- `input_folderA`: Path to input folder A containing images.
- `input_folderB`: Path to input folder B containing images.
- `output_folderA`: Path to output folder A for saving filtered images.
- `output_folderB`: Path to output folder B for saving filtered images.

## Resize Command
Command: `resize`

### Description
This command resizes images in two folders to specified dimensions.

### Arguments
- `dimensions`: Dimensions (x_dim, y_dim) for resizing.
  - Type: int
- `folderA`: Path to folder A containing images.
- `folderB`: Path to folder B containing images.

## Graph Command
Command: `graph`

### Description
This command generates graphs based on JSON data.

### Arguments
- `method`: Method for graph generation.
- `json`: Path to JSON file containing data for graph generation.
- `bottomcoverage`: The bottom percentage removed (i.e., don't plot the bottom 20% of the data).
- `topcoverage`: The top percentage removed.

## Ratio Command
Command: `ratio`

### Description
This command finds and prints all pairs of numbers with the same ratio that are smaller than the given numbers. Used to find other suitable resolutions that could be used.

### Arguments
- `num1`: The first number.
- `num2`: The second number.

## Average Command
Command: `average`

### Description
This command calculates the average of all PNG images in a specified folder and saves the result.

![image](https://github.com/AU-Bio-Inspired-Research-Group/pipelines/assets/121823795/0c50329c-a0bb-40e9-af50-4254143a9818)


### Arguments
- `inputFolder`: Path to the folder containing input images.
- `outputFile`: Filename for the output averaged image.
