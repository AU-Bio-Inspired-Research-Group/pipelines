import cv2
import numpy as np

def isolate_feature(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate absolute difference
    abs_diff = cv2.absdiff(image1, image2)
    
    # Thresholding
    _, mask = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Apply the mask to one of the original images
    isolated_feature = cv2.bitwise_and(image1, image1, mask=mask)
    
    return isolated_feature
# Example usage:
image1_path = "609A.png"
image2_path = "609.png"

isolated = isolate_feature(image1_path, image2_path)

# Display or save the isolated feature image
cv2.imshow('Isolated Feature', isolated)
cv2.waitKey(0)
cv2.destroyAllWindows()
