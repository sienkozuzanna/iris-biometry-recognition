import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects

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

def iris_brightness_1(grayscale_image):

    h, w = grayscale_image.shape
    roi = get_center_roi(grayscale_image, scale=0.4)
    
    mask = roi > 40  # ignoring black pixels (pupil and eyelashes)
    filtered_pixels = roi[mask]
    
    if filtered_pixels.size == 0:
        mean_brightness = np.mean(roi)
    else:
        mean_brightness = np.mean(filtered_pixels)
        
    return mean_brightness


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
def iris_binarization(grayscale_image, X_I, sharpen=True):
    if grayscale_image is None:
        raise ValueError("Image not found or unable to load.")
    if len(grayscale_image.shape) != 2:
        raise ValueError("Image must be a grayscale image.")
    
    sharpened = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(sharpened)
    enhanced = cv2.addWeighted(grayscale_image, 1.0, sharpened, 1.0, 0)
    grayscale_image=enhanced
    
    P = compute_binarization_threshold(grayscale_image)
    #binarization threshold for iris
    P_I = P / X_I 
    _, binary_iris = cv2.threshold(grayscale_image, P_I, 255, cv2.THRESH_BINARY)
    
    return binary_iris


#------------------------------------------------------------------------------------ IRIS recznie ------------------------------------------------------
def binarize_iris_manual(grayscale_image, brightness=None):
    if brightness is None:
        brightness=iris_brightness_1(grayscale_image)
    if brightness<95:
        xi = 1.95
    elif brightness>95 and brightness<96:
        xi=2.3
    elif brightness>96 and brightness<97:
        xi=1.8
    elif brightness>97 and brightness<98:
        xi=2.1
    elif brightness>98 and brightness<100:
        xi=2.0
    elif brightness>100 and brightness<102:
        xi=1.95
    elif brightness>102 and brightness<103:
        xi=1.9
    elif brightness>103 and brightness<104:
        xi=2.1
    elif brightness>104 and brightness<107:
        xi=1.97
    elif brightness>107 and brightness<108:
        xi=1.75
    elif brightness>108 and brightness<109:
        xi=1.8
    elif brightness>109 and brightness<111:
        xi=1.9
    elif brightness>111 and brightness<113:
        xi=1.85
    elif brightness>113 and brightness<116:
        xi=1.95
    elif brightness>116 and brightness<117:
        xi=1.7
    elif brightness>117.6 and brightness<118:
        xi=2
    elif brightness>117 and brightness<117.8:
        xi=1.5
    elif brightness>118 and brightness<119:
        xi=1.6
    elif brightness>119 and brightness<120:
        xi=2.1
    elif brightness>120 and brightness<121:
        xi=1.7
    elif brightness>121 and brightness<123:
        xi=1.8
    elif brightness>123.5 and brightness<124:
        xi=1.3
    elif brightness>123 and brightness<123.5:
        xi=1.8
    elif brightness>124 and brightness<127:
        xi=1.4
    elif brightness>127 and brightness<130:
        xi=2.25
    elif brightness>130 and brightness<135:
        xi=1.5
    elif brightness>135 and brightness<136:
        xi=2.15
    elif brightness>136 and brightness<140:
        xi=1.35
    elif brightness>140:
        xi=1.5
    
    binary_iris = iris_binarization(grayscale_image, xi)
    return binary_iris

# ------------------------------------------------------------------------------------------Pupil-------------------------------------------------
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

#----------------------------------------------------------------------------Iris------------------------------------------------------------------------

def clean_iris(grayscale_image, open_kernel_size1, close_kernel_size1, open_kernel_size2, close_kernel_size2,
                extra_cleaning=True, min_area=500,  horizontal_kernel_width=15):
    brightness = iris_brightness_1(grayscale_image)
    bin_image = binarize_iris_manual(grayscale_image, brightness)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size1, open_kernel_size1))  
    opened = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size1, close_kernel_size1))  
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)


    if extra_cleaning:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        eroded = cv2.erode(closed, erode_kernel, iterations=1)

        dilated = cv2.dilate(eroded, erode_kernel, iterations=1)
        blackhat = cv2.morphologyEx(closed, cv2.MORPH_BLACKHAT, erode_kernel)

        cleaned = cv2.bitwise_and(dilated, cv2.bitwise_not(blackhat))
        return cleaned
    kernel_erode_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1)) 
    eroded_horizontally = cv2.erode(cleaned, kernel_erode_horizontal, iterations=1)

    cleaned_bool = eroded_horizontally.astype(bool) 
    cleaned_filtered = remove_small_objects(cleaned_bool, min_size=min_area)
    cleaned_final = (cleaned_filtered * 255).astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size2, open_kernel_size2))  
    opened = cv2.morphologyEx(cleaned_final, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size2, close_kernel_size2))  
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed

def remove_eyelids_and_eyelashes(grayscale_image):
    brightness = iris_brightness_1(grayscale_image)
    bin_image = binarize_iris_manual(grayscale_image, brightness)
    
    cleaned_bool = bin_image.astype(bool)
    cleaned_filtered = remove_small_objects(cleaned_bool, min_size=500)
    cleaned_final = (cleaned_filtered * 255).astype(np.uint8)

    kernel_eyelash = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    no_eyelashes = cv2.morphologyEx(cleaned_final, cv2.MORPH_OPEN, kernel_eyelash)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    no_eyelids = cv2.morphologyEx(no_eyelashes, cv2.MORPH_OPEN, horizontal_kernel)

    contours, _ = cv2.findContours(no_eyelids, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height = grayscale_image.shape[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        if aspect_ratio > 4 and (y < height * 0.4 or y + h > height * 0.6):
            cv2.drawContours(no_eyelashes, [cnt], -1, 255, -1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    no_vertical_lines = cv2.morphologyEx(no_eyelashes, cv2.MORPH_CLOSE, vertical_kernel)

    cleaned_bool = no_vertical_lines.astype(bool)
    cleaned_filtered = remove_small_objects(cleaned_bool, min_size=100)
    cleaned_final = (cleaned_filtered * 255).astype(np.uint8)

    return cleaned_filtered


def iris_segmentation_pipeline(grayscale_image, pupil_X_P=5.2, iris_X_I=None):
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

def find_hough_circles(grayscale_image, dp=1.5, min_dist=30, param1=100, param2=30, min_radius=10, max_radius=80):
    brightness = iris_brightness_1(grayscale_image)
    bin_image = binarize_iris_manual(grayscale_image, brightness)

    circles = cv2.HoughCircles(bin_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2,
        minRadius=min_radius,maxRadius=max_radius)

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        return circles 
    else:
        return []

#---------------------------------------------------------------- Plots------------------------------------------------------------------

def plot_images_experiments(original_images, processed_images, n=3, figsize = (6,4)):
    n = min(n, len(original_images))
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