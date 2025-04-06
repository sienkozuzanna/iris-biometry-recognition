import cv2
import numpy as np
import matplotlib.pyplot as plt

#converting to grayscale
def convert_to_grayscale(image):
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compute_binarization_threshold(grayscale_image):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    h, w = grayscale_image.shape
    total_sum = np.sum(grayscale_image) #sum of all pixel values
    P = total_sum / (h * w) #mean brightness
    return P

#X_I from experiments 
def iris_binarization(grayscale_image, iris_mask, X_I):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    P = compute_binarization_threshold(grayscale_image)
    #binarization threshold for iris
    P_I = P / X_I 
    #applying iris mask
    iris_region = cv2.bitwise_and(grayscale_image, grayscale_image, mask=iris_mask)
    _, binary_iris = cv2.threshold(iris_region, P_I, 255, cv2.THRESH_BINARY)
    
    return binary_iris

#X_P from experiments
def pupil_binarization(grayscale_image, pupil_mask, X_P):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    P = compute_binarization_threshold(grayscale_image)
    #binarization threshold for pupil
    P_I = P / X_P
    #applying pupil mask
    pupil_region = cv2.bitwise_and(grayscale_image, grayscale_image, mask=pupil_mask)
    _, binary_pupil = cv2.threshold(pupil_region, P_I, 255, cv2.THRESH_BINARY)
    return binary_pupil
