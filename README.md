# Pipelines

To use pipelines clone the repository and cd into pipelines. Use `python pipelines.py [METHOD]` eg `python pipelines.py pipeline1 MAE doorA doorB outA outB`

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

## Filter Command
Command: `filter`

### Description
This command applies a filter to images in two input folders and saves the results in two output folders.

### Arguments
- `filter`: Filter to apply to images.
  - Choices: ["greyscale", "colour"]
- `input_folderA`: Path to input folder A containing images.
- `input_folderB`: Path to input folder B containing images.
- `output_folderA`: Path to output folder A for saving filtered images.
- `output_folderB`: Path to output folder B for saving filtered images.
