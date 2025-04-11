import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


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


def calculate_iris_threshold_mean_brightness(grayscale_image):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    mean_brightness = np.mean(grayscale_image)
    normalized_brightness = mean_brightness / 255.0
    print(f'normalized_brightness: {normalized_brightness:.3f}')
    
    # im jaśniejszy obraz, tym mniejsze X_I
    X_I = 1.9 - 0.5 * normalized_brightness
    
    return np.clip(X_I, 1.4, 1.9)

def get_center_roi(image, scale=0.5):
    h, w = image.shape
    dh = int(h * scale / 2)
    dw = int(w * scale / 2)
    center_h = h // 2
    center_w = w // 2
    return image[center_h - dh:center_h + dh, center_w - dw:center_w + dw]


def calculate_iris_threshold(grayscale_image):
    h, w = grayscale_image.shape
    roi = get_center_roi(grayscale_image, scale=0.4)
    
    mask = roi > 40  # ignoring black pixels (pupil and eyelashes)
    filtered_pixels = roi[mask]
    
    if filtered_pixels.size == 0:
        mean_brightness = np.mean(roi)
    else:
        mean_brightness = np.mean(filtered_pixels)
        
    normalized = mean_brightness / 255.0
    print(normalized)
    if normalized < 0.4:  # darker iris
        X_I = 1.8
    elif normalized > 0.5:  # lighter iris
        X_I = 1.6
    else:
        X_I = 1.8 - (normalized - 0.4) * 2.0 #interpolation
    return np.clip(X_I, 1.4, 1.9)

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

# morphology - open and close
def clean_pupil(grayscale_image, bin_X_P, open_kernel_size, close_kernel_size):
    bin_image = pupil_binarization(grayscale_image, bin_X_P)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))  
    opened = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))  
    final_pupil = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return final_pupil

# calculate pupil center an radius with projections
def pupil_center_radius(final_pupil):
    final_pupil = np.array(final_pupil)
    vertical_projection = np.sum(final_pupil ==0, axis=0)
    horizontal_projection = np.sum(final_pupil==0, axis=1)
    center_x = np.argmax(vertical_projection)
    center_y = np.argmax(horizontal_projection)

    pupil_mask =(final_pupil==0).astype(np.uint8)
    #radius
    y_ind, x_ind = np.where(pupil_mask==1) #najbardziej odległe punkty
    if len(x_ind) >0 and len(y_ind)>0:
        distances = np.sqrt((x_ind - center_x)**2 + (y_ind - center_y)**2) #odległość kazdego piksela od środka
        radius = np.percentile(distances, 85)
    else:
        radius =0
    
    return (center_x, center_y), radius

#calculate pupil center and radius with moments
def pupil_center_radius_moments(final_pupil):
    final_pupil = np.array(final_pupil)
    pupil_mask = (final_pupil ==0).astype(np.uint8) #same źrenice
    moments = cv2.moments(pupil_mask)
    if moments['m00'] != 0:
        center_x = int(moments['m10']/moments['m00'])
        center_y = int(moments['m01']/ moments['m00'])
    else: 
        (center_x,center_y),_ = pupil_center_radius(final_pupil)

    #radius
    y_ind, x_ind = np.where(pupil_mask==1) #najbardziej odległe punkty
    if len(x_ind) >0 and len(y_ind)>0:
        distances = np.sqrt((x_ind - center_x)**2 + (y_ind - center_y)**2) #odległość kazdego piksela od środka
        radius = np.max(distances)
    else:
        radius =0

    return (center_x,center_y), radius

# draw pupil center and circle
def draw_pupil_circle(final_pupil, center, radius):
    if len(final_pupil.shape) == 2:
        final_pupil = cv2.cvtColor(final_pupil, cv2.COLOR_GRAY2BGR)
    
    cv2.circle(final_pupil, center, radius, (0, 255, 0), 2)  
    cv2.drawMarker(final_pupil, center, (0, 0, 255), 
                   markerType=cv2.MARKER_CROSS, 
                   markerSize=10,
                   thickness=2)  
    return final_pupil

def clean_iris_region(iris_binary, pupil_center, pupil_radius):
    #removing eyelashes (vertical and horizontal sturctures rather this)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    temp = cv2.morphologyEx(iris_binary, cv2.MORPH_OPEN, kernel_vertical)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel_horizontal)

    #fixing holes
    kernel_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #small white dots
    iris_clean = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel_round) #small black holes
    iris_clean = cv2.morphologyEx(iris_clean, cv2.MORPH_CLOSE, kernel_round)

    #masking areas too far or too close to iris (eyeleads)
    mask = np.zeros_like(iris_clean)
    cv2.circle(mask, pupil_center, int(pupil_radius * 4), 255, -1)
    cv2.circle(mask, pupil_center, int(pupil_radius * 1.2), 0, -1)
    iris_clean = cv2.bitwise_and(iris_clean, mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(iris_clean)
    if num_labels > 1:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        iris_clean = (labels == largest_label).astype(np.uint8) * 255
    
    return iris_clean


def iris_segmentation_pipeline(grayscale_image, pupil_X_P=2.5, iris_X_I=None):
    pupil_binary = clean_pupil(grayscale_image, pupil_X_P, 5, 7)
    pupil_center, pupil_radius = pupil_center_radius_moments(pupil_binary)
    if iris_X_I is None:
        iris_X_I = calculate_iris_threshold(grayscale_image)
    iris_binary = iris_binarization(grayscale_image, iris_X_I)
    iris_clean = clean_iris_region(iris_binary, pupil_center, pupil_radius)
    #iris_center, iris_radius = detect_iris_boundary(iris_clean, pupil_center, pupil_radius)

    return {
        'pupil_center': pupil_center,
        'pupil_radius': pupil_radius,
        'pupil_binary': pupil_binary,
        'iris_binary': iris_clean
    }


def plot_images_experiments(original_images, processed_images, n=3, figsize = (6,4)):
    fig, axes = plt.subplots(n, 2, figsize=figsize)
    
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
