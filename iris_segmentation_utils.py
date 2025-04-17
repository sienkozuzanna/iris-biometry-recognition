import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
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
    #print(normalized)
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

def pupil_pipeline(grayscale_image, X_P=5.2, open_kernel_size=7, close_kernel_size=9):
    clean_pupil_image = clean_pupil(grayscale_image, X_P, open_kernel_size, close_kernel_size)
    center, radius = pupil_center_radius_moments(clean_pupil_image)
    
    return clean_pupil_image, center, radius

#----------------------------------------------------------------------------Iris------------------------------------------------------------------------

def clean_iris(grayscale_image, open_kernel_size1=5, close_kernel_size1=5, 
               open_kernel_size2=5, close_kernel_size2=5,
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

    kernel_close_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close_big)

    kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open_small)

    return opened


def remove_eyelids_and_eyelashes(grayscale_image):
    brightness = iris_brightness_1(grayscale_image)
    bin_image = binarize_iris_manual(grayscale_image, brightness)
    
    cleaned_bool = bin_image.astype(bool)
    cleaned_filtered = remove_small_objects(cleaned_bool, min_size=500)
    cleaned_final = (cleaned_filtered * 255).astype(np.uint8)

    kernel_eyelash = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    no_eyelashes = cv2.morphologyEx(cleaned_final, cv2.MORPH_OPEN, kernel_eyelash)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    no_eyelids = cv2.morphologyEx(no_eyelashes, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 40))
    no_vertical_lines = cv2.morphologyEx(no_eyelids, cv2.MORPH_OPEN, vertical_kernel)
    
    contours, _ = cv2.findContours(no_eyelids, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height = grayscale_image.shape[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        if aspect_ratio > 4 and (y < height * 0.2 or y + h > height * 0.3):
            cv2.drawContours(no_eyelashes, [cnt], -1, 255, -1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    no_vertical_lines = cv2.morphologyEx(no_eyelashes, cv2.MORPH_CLOSE, vertical_kernel)

    cleaned_bool = no_vertical_lines.astype(bool)
    cleaned_filtered = remove_small_objects(cleaned_bool, min_size=100)
    cleaned_final = (cleaned_filtered * 255).astype(np.uint8)

    return cleaned_final

def detect_canny_edges(grayscale_image, low_threshold=50, high_threshold=150):
    brightness = iris_brightness_1(grayscale_image)
    bin_image = binarize_iris_manual(grayscale_image, brightness)
    edges=cv2.Canny(bin_image, low_threshold, high_threshold)
    return edges

def iris_morphology(bin_image, open_kernel_size=9, close_kernel_size=3, 
                    remove_horizontal_lines=True, remove_vertical_lines=True):
    
    bin_image-remove_small_objects_skimage(bin_image, min_size=500)

    if remove_horizontal_lines:
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel_horizontal)
    if remove_vertical_lines:
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel_vertical)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))  
    opened = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    final=remove_small_objects_skimage(closed, min_size=500)
    return final

def remove_small_objects_skimage(bin_image, min_size=500):
    bin_image = (bin_image > 0)
    cleaned = remove_small_objects(bin_image, min_size=min_size)
    return (cleaned * 255).astype(np.uint8)

def extract_iris_region(image, pupil_center, pupil_radius, iris_radius):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, tuple(map(int, pupil_center)), int(iris_radius), 255, thickness=-1)
    cv2.circle(mask, tuple(map(int, pupil_center)), int(pupil_radius), 0, thickness=-1)
    if len(image.shape) == 3:
        result = cv2.bitwise_and(image, image, mask=mask)
    else:
        result = cv2.bitwise_and(image, mask)

    return result


def iris_pipeline(grayscale_image):
    brightness = iris_brightness_1(grayscale_image)
    bin_image = binarize_iris_manual(grayscale_image, brightness)
    cleaned_image=iris_morphology(bin_image)
    low_threshold, high_threshold=10,120
    canny_edges=cv2.Canny(cleaned_image, low_threshold, high_threshold)

    clean_pupil_image, pupil_center, pupil_radius = pupil_pipeline(grayscale_image)

    circles=find_hough_circles(canny_edges, pupil_center, pupil_radius)
    print("Detected circles:", circles)
    image_with_circles = draw_circles(grayscale_image.copy(), circles, pupil_center, pupil_radius)
    if circles:
        iris_radius = circles[0][2]
        extracted_iris = extract_iris_region(grayscale_image, pupil_center, pupil_radius, iris_radius)
        unwrapped_iris = unwrap_iris(extracted_iris, pupil_center, pupil_radius, iris_radius)
    else:
        extracted_iris = np.zeros_like(grayscale_image)
        unwrapped_iris = np.zeros((64, 360), dtype=np.uint8)

    return canny_edges, cleaned_image, image_with_circles, circles, extracted_iris, unwrapped_iris


def find_hough_circles(bin_image, pupil_center, pupil_radius, dp=1.5, min_dist=30,
                        param1=100, param2=30, max_radius=80, max_distance=30):
    min_radius=int(pupil_radius*1.35)
    circles = cv2.HoughCircles(bin_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, 
                               param2=param2, minRadius=min_radius, maxRadius=max_radius)
    detected_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        for (x, y, r) in circles:
            distance = np.sqrt((x - pupil_center[0])**2 + (y - pupil_center[1])**2)

            if distance <= max_distance:
                new_circle = (int(pupil_center[0]), int(pupil_center[1]), r)
                detected_circles.append(new_circle)

    return detected_circles

def find_hough_circles_all(bin_image, dp=1.5, min_dist=30,
                            param1=100, param2=30, min_radius=0, max_radius=80):
    circles = cv2.HoughCircles(bin_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1,
        param2=param2,minRadius=min_radius,maxRadius=max_radius)

    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            detected_circles.append((x, y, r))
    return detected_circles


def draw_circles(image, circles, pupil_center=None, pupil_radius=None):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.drawMarker(image, (x, y), (0, 0, 255),   
                       markerType=cv2.MARKER_CROSS, 
                       markerSize=10,
                       thickness=2)
        
    if pupil_center is not None and pupil_radius is not None:
        cv2.circle(image, tuple(map(int, pupil_center)), int(pupil_radius), (255, 0, 0), 2)

    return image

def unwrap_iris(image, pupil_center, pupil_radius, iris_radius):
    rad_res, ang_res =64, 360 #number of radial sampling steps, number of angular sampling steps
    unwrapped = np.zeros((rad_res, ang_res), dtype=np.uint8)
    center_x, center_y = pupil_center

    for i in range(rad_res):
        r = pupil_radius + (iris_radius - pupil_radius) * (i / rad_res)
        for j in range(ang_res):
            theta = 2 * np.pi * (j / ang_res)
            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))

            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                unwrapped[i, j] = image[y, x]
            else:
                unwrapped[i, j] = 0  #outside iris

    return unwrapped


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

def plot_iris_stages(original_images, cleaned_images, canny_edges, images_with_circles, n=3, figsize=(12, 6)):
    n = min(n, len(original_images))
    fig, axes = plt.subplots(n, 4, figsize=figsize)

    for i in range(n):
        axes[i, 0].imshow(original_images.iloc[i], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original')

        axes[i, 1].imshow(cleaned_images.iloc[i], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Cleaned')

        axes[i, 2].imshow(canny_edges.iloc[i], cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Canny Edges')

        axes[i, 3].imshow(images_with_circles.iloc[i])
        axes[i, 3].axis('off')
        axes[i, 3].set_title('Detected Circles')

    plt.tight_layout()
    plt.show()
