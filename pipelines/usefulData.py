import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def analyze_results(results_file, method):
    # Load results from the JSON file
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Extract the specified method values
    method_values = {img: data[method] for img, data in results.items() if method in data}
    
    if not method_values:
        print(f"No data found for method '{method}'")
        return
    
    # Convert to a pandas DataFrame for easier analysis
    df = pd.DataFrame(list(method_values.items()), columns=['Image', method])
    
    # Find the image with the highest and lowest value
    max_value = df[method].max()
    min_value = df[method].min()
    max_image = df[df[method] == max_value]['Image'].values[0]
    min_image = df[df[method] == min_value]['Image'].values[0]
    
    print(f"Image with highest {method} value: {max_image} ({max_value})")
    print(f"Image with lowest {method} value: {min_image} ({min_value})")
    
    # Calculate mean and standard deviation
    mean_value = df[method].mean()
    std_dev = df[method].std()
    
    print(f"Mean {method} value: {mean_value}")
    print(f"Standard deviation of {method} values: {std_dev}")
    
    # Create a box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[method], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Box Plot of {method} Values')
    plt.xlabel(method)
    plt.show()

def average_images(directory, output_filename):
    # Access all PNG files in the specified directory
    allfiles = os.listdir(directory)
    imlist = [filename for filename in allfiles if filename.lower().endswith(".png")]

    # Check if there are any PNG files in the directory
    if not imlist:
        print("No PNG files found in the directory.")
        return

    # Assuming all images are the same size, get dimensions of the first image
    w, h = Image.open(os.path.join(directory, imlist[0])).size
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = np.array(Image.open(os.path.join(directory, im)), dtype=float)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save, and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save(output_filename)
    out.show()