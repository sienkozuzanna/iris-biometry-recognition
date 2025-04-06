import cv2
import numpy as np
import matplotlib.pyplot as plt

#converting to grayscale
def convert_to_grayscale(image):
    if image is None:
        raise ValueError("Image not found or unable to load.")
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        return image_np

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
def iris_binarization(grayscale_image, X_I):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    P = compute_binarization_threshold(grayscale_image)
    #binarization threshold for iris
    P_I = P / X_I 
    _, binary_iris = cv2.threshold(grayscale_image, P_I, 255, cv2.THRESH_BINARY)
    
    return binary_iris

#X_P from experiments
def pupil_binarization(grayscale_image, X_P):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    P = compute_binarization_threshold(grayscale_image)
    #binarization threshold for pupil
    P_P = P / X_P
    _, binary_pupil = cv2.threshold(grayscale_image, P_P, 255, cv2.THRESH_BINARY)
    return binary_pupil


def plot_images_experiments(original_images, processed_images, n=3):
    fig, axes = plt.subplots(n, 2, figsize=(6, 2 * n))
    
    for i in range(n):
        axes[i, 0].imshow(original_images.iloc[i], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Original Image {i+1}')
        
        axes[i, 1].imshow(processed_images.iloc[i], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Processed Image {i+1}')

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.show()

