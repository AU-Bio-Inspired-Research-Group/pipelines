import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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