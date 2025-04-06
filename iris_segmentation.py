import cv2
import numpy as np
import matplotlib.pyplot as plt

#converting to grayscale
def convert_to_grayscale(image):
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#binarization for custom threshold
def binary_threshold(image, threshold=127):
    if image is None:
        raise ValueError("Image not found or unable to load.")
    if threshold < 0 or threshold > 255:
        raise ValueError("Threshold must be between 0 and 255.")
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image
