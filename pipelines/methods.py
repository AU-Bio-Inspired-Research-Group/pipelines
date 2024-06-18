import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return float(err)  # Ensure MSE is converted to float

# Function to calculate Peak Signal-to-Noise Ratio (PSNR)
def psnr(imageA, imageB):
    mse_val = mse(imageA, imageB)
    if mse_val == 0:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse_val)))  # Ensure PSNR is converted to float

# Function to calculate Mean Absolute Error (MAE)
def mae(imageA, imageB):
    return float(np.mean(np.abs(imageA - imageB)))  # Ensure MAE is converted to float

# Function to calculate Normalized Cross-Correlation (NCC)
def ncc(imageA, imageB):
    num = np.sum((imageA - np.mean(imageA)) * (imageB - np.mean(imageB)))
    denom = np.sqrt(np.sum((imageA - np.mean(imageA)) ** 2) * np.sum((imageB - np.mean(imageB)) ** 2))
    return float(num / denom)  # Ensure NCC is converted to float

def ssim(img1, img2, gaussian_weights=True, sigma=1.5):
    # Convert images to grayscale
    gray1 = io.imread(img1, as_gray=True)
    gray2 = io.imread(img2, as_gray=True)
    # Compute SSIM
    score = ssim(gray1, gray2, data_range=gray2.max() - gray2.min(), gaussian_weights=gaussian_weights, sigma=sigma)
    print(score)
    return score

# Function to calculate Histogram Intersection
def histogram_intersection(imageA, imageB):
    histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    return float(cv2.compareHist(histA, histB, cv2.HISTCMP_INTERSECT))  # Ensure Histogram Intersection is converted to float

# Function to calculate Earth Mover's Distance (EMD)
def emd(imageA, imageB):
    histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    return float(cv2.EMD(histA, histB, cv2.DIST_L2))  # Ensure EMD is converted to float

# Function to calculate the Absolute Difference
def abs_diff(imageA, imageB):
    return float(np.sum(np.abs(imageA.astype(np.float32) - imageB.astype(np.float32))))  # Ensure Absolute Difference is converted to float

# Function to calculate Correlation Coefficient
def correlation(imageA, imageB):
    return float(cv2.matchTemplate(imageA, imageB, cv2.TM_CCORR_NORMED)[0][0])  # Ensure Correlation Coefficient is converted to float

# Function to calculate Bhattacharyya Distance
def bhattacharyya(imageA, imageB):
    histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    return float(cv2.compareHist(histA, histB, cv2.HISTCMP_BHATTACHARYYA))  # Ensure Bhattacharyya Distance is converted to float

# New function to calculate color histogram comparison
def color_histogram_comparison(imageA, imageB):
    histA = cv2.calcHist([imageA], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imageB], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)
    return float(cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL))  # Ensure Color Histogram Comparison is converted to float

def correlation(imageA, imageB):
    return float(cv2.matchTemplate(imageA, imageB, cv2.TM_CCORR_NORMED)[0][0]) 