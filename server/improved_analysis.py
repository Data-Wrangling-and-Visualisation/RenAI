import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import time
import random
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from skimage import feature, color, exposure, util, filters
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.cluster import KMeans

ENHANCED_COLOR_ANALYSIS_AVAILABLE = True

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    import colorsys
    try:
        from colormath.color_objects import sRGBColor, LabColor
        from colormath.color_conversions import convert_color
        from colormath.color_diff import delta_e_cie2000
    except ImportError:
        print("Colormath library not found. Full color analysis features will be limited.")
        ENHANCED_COLOR_ANALYSIS_AVAILABLE = False
except ImportError as e:
    ENHANCED_COLOR_ANALYSIS_AVAILABLE = False
    print(f"Enhanced color analysis libraries (sklearn/kneed/colorsys) not available: {e}")

ENHANCED_STYLE_ANALYSIS_AVAILABLE = True
ENHANCED_COMPOSITION_ANALYSIS_AVAILABLE = True

# Функция улучшенного анализа стиля
def enhanced_style_analysis(img_pil, embedding=None):
    """
    Performs an enhanced style analysis of an image.
    
    Args:
        img_pil: PIL image for analysis
        embedding: pre-calculated embedding (optional)
        
    Returns:
        dict: Dictionary with style parameters
    """
    try:
        if img_pil is None:
            print("Error: received empty image (None)")
            return {
                'linearity': 0.5,
                'colorfulness': 0.5,
                'complexity': 0.5,
                'contrast': 0.5,
                'symmetry': 0.5,
                'texture': 0.5,
                'is_fallback': True
            }
            
        start_time = time.time()
        print(f"Starting style analysis of image size {img_pil.size}")
        
        if img_pil.mode != 'RGB':
            print(f"Conversion of image from {img_pil.mode} to RGB format")
            img_pil = img_pil.convert('RGB')
        
        img_np = np.array(img_pil)
        print(f"Converted to numpy array size {img_np.shape}")
        
        if len(img_np.shape) < 3:
            print("Error: image is not RGB (not enough channels)")
            return {
                'linearity': 0.5,
                'colorfulness': 0.5,
                'complexity': 0.5,
                'contrast': 0.5,
                'symmetry': 0.5,
                'texture': 0.5,
                'is_fallback': True
            }
            
        if img_np.shape[2] != 3:
            print(f"Error: incorrect number of channels ({img_np.shape[2]}), requires 3")
            return {
                'linearity': 0.5,
                'colorfulness': 0.5,
                'complexity': 0.5,
                'contrast': 0.5,
                'symmetry': 0.5,
                'texture': 0.5,
                'is_fallback': True
            }
            
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        print(f"Image converted to grayscale size {img_gray.shape}")
        
        print("Linearity analysis...")
        edges = cv2.Canny(img_gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                              minLineLength=20, maxLineGap=10)
        
        linearity_score = 0.0
        if lines is not None:
            linearity_score = min(1.0, len(lines) / (img_gray.shape[0] * img_gray.shape[1] * 0.01))
            print(f"Found {len(lines)} lines, linearity: {linearity_score:.4f}")
        else:
            print("Lines not found")
        
        print("Colorfulness analysis...")
        r, g, b = cv2.split(img_np)
        
        rg = r.astype(np.int32) - g.astype(np.int32)
        yb = 0.5 * (r.astype(np.int32) + g.astype(np.int32)) - b.astype(np.int32)
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        rg_mean = np.abs(np.mean(rg))
        yb_mean = np.abs(np.mean(yb))
        
        colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
        colorfulness_score = min(1.0, colorfulness / 150.0)
        print(f"Colorfulness: {colorfulness:.2f}, normalized score: {colorfulness_score:.4f}")
        
        print("Complexity analysis...")
        complexity_entropy = shannon_entropy(img_gray)
        complexity_score = min(1.0, complexity_entropy / 8.0)  # 8 - макс энтропия для 8-бит изображений
        print(f"Entropy: {complexity_entropy:.4f}, normalized complexity: {complexity_score:.4f}")
        
        print("Анализ контраста...")
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Нормализация
        cumsum = np.cumsum(hist)
        p10 = np.argmax(cumsum >= 0.1)
        p90 = np.argmax(cumsum >= 0.9)
        contrast_score = min(1.0, (p90 - p10) / 255.0)
        print(f"P10: {p10}, P90: {p90}, контраст: {contrast_score:.4f}")
        
        print("Symmetry analysis...")
        h, w = img_gray.shape
        
        left_half = img_gray[:, :w//2]
        right_half = cv2.flip(img_gray[:, w//2:], 1)
        
        min_w = min(left_half.shape[1], right_half.shape[1])
        h_diff = cv2.absdiff(left_half[:, :min_w], right_half[:, :min_w])
        h_symmetry = 1.0 - np.mean(h_diff) / 255.0
        
        top_half = img_gray[:h//2, :]
        bottom_half = cv2.flip(img_gray[h//2:, :], 0)
        
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        v_diff = cv2.absdiff(top_half[:min_h, :], bottom_half[:min_h, :])
        v_symmetry = 1.0 - np.mean(v_diff) / 255.0
        
        symmetry_score = (0.7 * h_symmetry + 0.3 * v_symmetry)
        print(f"Horizontal symmetry: {h_symmetry:.4f}, vertical: {v_symmetry:.4f}, total: {symmetry_score:.4f}")
        
        print("Texture analysis...")
        if max(img_gray.shape) > 300:
            scale_factor = 300 / max(img_gray.shape)
            img_gray_resized = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor, 
                                        interpolation=cv2.INTER_AREA)
            print(f"Image downsized for texture analysis: {img_gray_resized.shape}")
        else:
            img_gray_resized = img_gray
            
        # Quantization for efficiency
        bins = 32  # 32 gray levels
        img_gray_quantized = np.digitize(img_gray_resized, np.linspace(0, 255, bins+1)) - 1
        
        try:
            distances = [1, 3]  # Consider pixels at distances 1 and 3
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
            
            glcm = graycomatrix(img_gray_quantized, distances, angles, bins, symmetric=True, normed=True)
            
            contrast_texture = graycoprops(glcm, 'contrast').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            print(f"GLCM properties - contrast: {contrast_texture:.4f}, homogeneity: {homogeneity:.4f}, energy: {energy:.4f}, correlation: {correlation:.4f}")
            
            texture_score = 0.5 * contrast_texture + 0.2 * (1-homogeneity) + 0.2 * (1-energy) + 0.1 * correlation
            texture_score = min(1.0, max(0.0, texture_score / 50.0))
            print(f"Texture score: {texture_score:.4f}")
        except Exception as e:
            print(f"Error in texture analysis, using fallback: {e}")
            texture_score = min(1.0, np.std(img_gray) / 50.0)
            print(f"Fallback texture score (based on standard deviation): {texture_score:.4f}")
        
        style_values = [
            min(0.95, max(0.05, linearity_score)),
            min(0.95, max(0.05, colorfulness_score)),
            min(0.95, max(0.05, complexity_score)),
            min(0.95, max(0.05, contrast_score)),
            min(0.95, max(0.05, symmetry_score)),
            min(0.95, max(0.05, texture_score))
        ]
        
        end_time = time.time()
        print(f"Style analysis took {end_time - start_time:.2f} seconds")
        print(f"Final style values: {style_values}")
        
        return {
            'linearity': style_values[0],
            'colorfulness': style_values[1],
            'complexity': style_values[2],
            'contrast': style_values[3],
            'symmetry': style_values[4],
            'texture': style_values[5],
            'is_fallback': False
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Critical error in enhanced_style_analysis: {e}")
        return {
            'linearity': 0.5,
            'colorfulness': 0.5,
            'complexity': 0.5,
            'contrast': 0.5,
            'symmetry': 0.5,
            'texture': 0.5,
            'is_fallback': True
        }


# Функция улучшенного анализа композиции
def enhanced_composition_analysis(img_pil):
    """
    Performs an enhanced composition analysis of an image.
    
    Args:
        img_pil: PIL image for analysis
        
    Returns:
        dict: Dictionary with composition parameters
    """
    try:
        start_time = time.time()
        
        # Convert image
        img_np = np.array(img_pil)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = img_gray.shape
        
        # 1. SYMMETRY - More accurate calculation
        # Horizontal symmetry with weights
        left_half = img_gray[:, :w//2]
        right_half = cv2.flip(img_gray[:, w//2:], 1)
        
        min_w = min(left_half.shape[1], right_half.shape[1])
        
        # Create a weight mask (pixels closer to the center are more important)
        weight_mask = np.linspace(0.5, 1.0, min_w)
        weight_mask = np.tile(weight_mask, (left_half.shape[0], 1))
        
        h_diff = cv2.absdiff(left_half[:, :min_w], right_half[:, :min_w])
        weighted_h_diff = h_diff * weight_mask
        h_symmetry = 1.0 - np.sum(weighted_h_diff) / (np.sum(weight_mask) * 255)
        
        # 2. RULE OF THIRDS - Improved algorithm with gradients
        third_h, third_v = h // 3, w // 3
        
        # Create grid points for the rule of thirds
        grid_points = [
            (third_v, third_h),         # Upper left
            (third_v * 2, third_h),     # Upper right
            (third_v, third_h * 2),     # Lower left
            (third_v * 2, third_h * 2)  # Lower right
        ]
        
        # Calculate gradients
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradients
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude) if np.max(gradient_magnitude) > 0 else gradient_magnitude
        
        # Check if important elements match the rule of thirds points
        roi_score = 0
        roi_size = int(min(h, w) * 0.1)  # 10% of image size
        
        for point in grid_points:
            x, y = point
            x1, y1 = max(0, x - roi_size // 2), max(0, y - roi_size // 2)
            x2, y2 = min(w, x + roi_size // 2), min(h, y + roi_size // 2)
            
            if x1 < x2 and y1 < y2:  # Check valid ROI size
                roi = gradient_magnitude[y1:y2, x1:x2]
                if roi.size > 0:
                    # If there are strong gradients in the ROI, increase the score
                    roi_score += min(0.25, np.mean(roi) * 5.0)  # Maximum 0.25 per point
        
        # 3. LEADING LINES - Use improved Hough algorithm
        edges = cv2.Canny(img_gray, 100, 200)
        
        # Hough lines give us directions
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                              minLineLength=min(h, w)//8, maxLineGap=20)
        
        leading_lines_score = 0
        if lines is not None and len(lines) > 0:
            # Calculate line angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                    angles.append(angle)
            
            # If there are dominant directions, this indicates leading lines
            if angles:
                # Group similar angles (within 10 degrees)
                angle_groups = {}
                for angle in angles:
                    key = round(angle / 10) * 10
                    if key in angle_groups:
                        angle_groups[key] += 1
                    else:
                        angle_groups[key] = 1
                
                # If there is a strong grouping of angles, this indicates leading lines
                dominant_angles = sorted(angle_groups.items(), key=lambda x: x[1], reverse=True)
                if dominant_angles and dominant_angles[0][1] >= 3:  # If at least 3 lines match the angle
                    leading_lines_score = min(1.0, dominant_angles[0][1] / 10.0)
        
        # 4. DEPTH - Analysis of perspective and gradients
        # Calculate brightness gradient from bottom to top (usually brighter objects in front)
        bottom_half_mean = np.mean(img_gray[h//2:, :])
        top_half_mean = np.mean(img_gray[:h//2, :])
        
        # Difference in brightness between bottom and top halves can indicate depth
        brightness_gradient = (bottom_half_mean - top_half_mean) / 255.0
        # Normalize to interval [0,1] and consider it may be negative
        brightness_gradient_score = min(1.0, max(0.0, (brightness_gradient + 0.5) / 1.0))
        
        # Additional: check blur in different parts of the image
        # Background objects are often more blurred
        foreground = img_gray[:h//3*2, :]  # Bottom 2/3 usually foreground
        background = img_gray[:h//3, :]    # Top 1/3 usually background
        
        foreground_laplacian = cv2.Laplacian(foreground, cv2.CV_64F).var() if foreground.size > 0 else 0
        background_laplacian = cv2.Laplacian(background, cv2.CV_64F).var() if background.size > 0 else 0
        
        # If foreground is sharper than background, this indicates depth
        focus_diff = max(0, foreground_laplacian - background_laplacian) / (foreground_laplacian + 1e-5)
        focus_score = min(1.0, focus_diff)
        
        # Combine depth scores
        depth_score = 0.6 * brightness_gradient_score + 0.4 * focus_score
        
        # 5. FRAMING - Check edges around the image
        border_width = int(min(h, w) * 0.1)  # 10% of image size
        
        # Create mask only for edges
        inner_mask = np.zeros_like(img_gray)
        cv2.rectangle(inner_mask, (border_width, border_width), (w-border_width, h-border_width), 255, border_width)
        border_region = cv2.bitwise_and(img_gray, inner_mask)
        
        # Calculate edge intensity in the framing area
        border_edges = cv2.Canny(border_region, 100, 200)
        
        # Total pixel counter for the perimeter
        total_border_pixels = 4 * (w + h - 2 * border_width) * border_width
        if total_border_pixels > 0:
            framing_score = min(1.0, np.sum(border_edges > 0) / total_border_pixels * 5.0)  # *5 for scaling
        else:
            framing_score = 0.5  # Fallback value
        
        # 6. BALANCE - Evaluate the distribution of visual weight
        # Split the image into quarters
        top_left = img_gray[:h//2, :w//2]
        top_right = img_gray[:h//2, w//2:]
        bottom_left = img_gray[h//2:, :w//2]
        bottom_right = img_gray[h//2:, w//2:]
        
        # Calculate visual weight for each quarter (combination of brightness and complexity)
        weights = []
        for quad in [top_left, top_right, bottom_left, bottom_right]:
            if quad.size > 0:
                quad_edges = cv2.Canny(quad, 100, 200)
                # Combine average brightness and edge density for visual weight
                edge_density = np.sum(quad_edges > 0) / quad.size
                quad_weight = np.mean(quad) / 255.0 + edge_density * 5.0  # *5 for increasing edge influence
                weights.append(quad_weight)
            else:
                weights.append(0)
        
        # Ideal balance: low weight variance
        if weights:
            weight_variance = np.var(weights)
            balance_score = 1.0 - min(1.0, weight_variance * 2.0)  # Invert and scale
        else:
            balance_score = 0.5  # Fallback value
        
        # Final composition values
        composition_scores = [
            min(0.95, max(0.05, h_symmetry)),
            min(0.95, max(0.05, roi_score)),
            min(0.95, max(0.05, leading_lines_score)),
            min(0.95, max(0.05, depth_score)),
            min(0.95, max(0.05, framing_score)),
            min(0.95, max(0.05, balance_score))
        ]
        
        end_time = time.time()
        print(f"Composition analysis took {end_time - start_time:.2f} seconds")
        
        # Return dictionary instead of array
        return {
            'symmetry': composition_scores[0],
            'rule_of_thirds': composition_scores[1],
            'leading_lines': composition_scores[2],
            'depth': composition_scores[3],
            'framing': composition_scores[4],
            'balance': composition_scores[5],
            'is_fallback': False
        }
    except Exception as e:
        print(f"Error in enhanced_composition_analysis: {e}")
        return {
            'symmetry': 0.5,
            'rule_of_thirds': 0.5,
            'leading_lines': 0.5,
            'depth': 0.5,
            'framing': 0.5,
            'balance': 0.5,
            'is_fallback': True
        }


# Функция улучшенного анализа цвета
def enhanced_color_analysis(img):
    """
    Performs an enhanced color palette analysis.
    
    Args:
        img: PIL image object.
        
    Returns:
        dict: Dictionary with color analysis results.
    """
    try:
        logger.info("Starting enhanced_color_analysis.")
        # 1. Check for monochrome
        if is_monochrome_image(img):
            logger.info("Image detected as monochrome. Running monochrome analysis.")
            return analyze_monochrome_image(img)

        logger.info("Image is color. Proceeding with color analysis.")
        # 2. Prepare data
        img_rgb = img.convert('RGB')
        img_np = np.array(img_rgb)
        pixels = img_np.reshape(-1, 3)

        # Removing almost black and almost white pixels for clustering
        threshold = 10 # Threshold for black
        white_threshold = 245 # Threshold for white
        mask = (np.mean(pixels, axis=1) > threshold) & (np.mean(pixels, axis=1) < white_threshold)
        pixels_filtered = pixels[mask]

        if len(pixels_filtered) < 50: # Need enough pixels
            logger.warning("Not enough filtered pixels for robust clustering. Using all pixels.")
            pixels_filtered = pixels
        elif len(pixels_filtered) > 50000: # Too many pixels, sample
            logger.info(f"Sampling {len(pixels_filtered)} pixels down to 50000 for clustering.")
            indices = np.random.choice(len(pixels_filtered), 50000, replace=False)
            pixels_filtered = pixels_filtered[indices]

        if len(pixels_filtered) == 0:
            logger.error("No pixels left after filtering for clustering. Falling back.")
            raise ValueError("No pixels to cluster after filtering.")

        logger.info(f"Determining optimal clusters for {len(pixels_filtered)} pixels...")
        # 3. Determine optimal number of clusters
        n_clusters = determine_optimal_clusters(pixels_filtered)
        logger.info(f"Optimal number of clusters: {n_clusters}")

        # 4. K-Means clustering
        logger.info("Performing K-Means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels_filtered)
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # 5. Calculate percentages for each cluster
        logger.info("Calculating cluster percentages...")
        counts = np.bincount(labels)
        total_pixels = len(labels)
        percentages = (counts / total_pixels) * 100

        # Ensure we have centers and percentages
        if len(centers) == 0 or len(percentages) == 0:
            logger.error("KMeans resulted in zero clusters. Falling back.")
            raise ValueError("KMeans resulted in zero clusters.")

        # Sort by percentages
        sorted_indices = np.argsort(percentages)[::-1]
        centers = centers[sorted_indices]
        percentages = percentages[sorted_indices]

        # Limit the maximum number of colors for display/analysis
        max_colors_for_analysis = 8
        if len(centers) > max_colors_for_analysis:
            logger.info(f"Reducing clusters from {len(centers)} to {max_colors_for_analysis} for analysis.")
            centers = centers[:max_colors_for_analysis]
            percentages = percentages[:max_colors_for_analysis]
            # Recalculate percentages so they sum to 100%
            percentages = (percentages / percentages.sum()) * 100

        logger.info("Analyzing color properties (getting HSV values)...")
        # 6. Get HSV colors from cluster centers
        hsv_colors = analyze_color_properties(centers)

        # Check if hsv_colors is empty
        if not hsv_colors:
            logger.error("Could not get HSV colors. Falling back.")
            raise ValueError("Failed to derive HSV colors from cluster centers.")

        logger.info("Determining color harmony...")
        # 7. Determine color harmony
        color_harmony = determine_color_harmony(hsv_colors)

        logger.info("Determining emotional impact...")
        # 8. Determine emotional impact
        emotional_impact = determine_emotional_impact(hsv_colors, percentages)

        # 9. Formatting chart data
        logger.info("Formatting chart data...")
        chart_data = []
        for i, center in enumerate(centers):
            hex_color = '#{:02x}{:02x}{:02x}'.format(center[0], center[1], center[2])
            name = get_closest_color_name(tuple(center))
            chart_data.append({
                "name": name,
                "color": hex_color,
                "value": round(percentages[i], 2)
            })

        logger.info("Enhanced color analysis successful.")
        return {
            "dominant_colors": [{"color": '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]), "percentage": round(p, 2)} for c, p in zip(centers, percentages)],
            "chart_data": chart_data,
            # Remove color_properties, its components are now separate fields
            "color_harmony": color_harmony,
            "emotional_impact": emotional_impact,
            "is_legacy": False,
            "is_monochrome": False
        }

    except Exception as e:
        import traceback
        logger.error(f"Error during enhanced_color_analysis: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Falling back to legacy_color_analysis.")
        # Return result of the old function in case of an error
        return legacy_color_analysis(img)

def is_monochrome_image(img):
    """Determines if the RGB image is monochrome (black and white)"""
    img_array = np.array(img)
    # Check if the values of R, G and B are the same for each pixel
    return np.all(img_array[:,:,0] == img_array[:,:,1]) and np.all(img_array[:,:,1] == img_array[:,:,2])

def analyze_monochrome_image(img):
    """Analyzes monochrome image"""
    # Convert to grayscale if not already
    if img.mode != 'L':
        gray_img = img.convert('L')
    else:
        gray_img = img
    
    # Get grayscale histogram
    histogram = gray_img.histogram()
    total_pixels = sum(histogram)
    
    # Split into 5 ranges: very dark, dark, medium, light, very light
    ranges = [
        (0, 50, "Very Dark"),
        (51, 100, "Dark"),
        (101, 150, "Medium"),
        (151, 200, "Light"),
        (201, 255, "Very Light")
    ]
    
    dominant_grays = []
    labels = []
    percentages = []
    hex_colors = []
    
    for start, end, name in ranges:
        count = sum(histogram[start:end+1])
        percentage = (count / total_pixels) * 100
        
        if percentage >= 1.0:  # Only if percentage is significant
            # Average shade in the range
            avg_shade = (start + end) // 2
            hex_code = '#{:02x}{:02x}{:02x}'.format(avg_shade, avg_shade, avg_shade)
            
            # Determine brightness and saturation for monochrome colors
            brightness = avg_shade / 255.0
            saturation = 0.0  # Monochrome colors have zero saturation
            
            dominant_grays.append({
                "rgb": [avg_shade, avg_shade, avg_shade],
                "hex": hex_code,
                "percentage": round(percentage, 1),
                "name": name,
                "brightness": round(brightness, 2),
                "saturation": round(saturation, 2),
                "is_dark": avg_shade < 128
            })
            
            labels.append(name)
            percentages.append(round(percentage, 1))
            hex_colors.append(hex_code)
    
    # Sort by percentage (descending)
    dominant_grays.sort(key=lambda x: x["percentage"], reverse=True)
    
    # Create chart data
    chart_data = {
        "labels": [item["name"] for item in dominant_grays],
        "datasets": [{
            "data": [item["percentage"] for item in dominant_grays],
            "backgroundColor": [item["hex"] for item in dominant_grays],
            "borderColor": ["#ffffff"] * len(dominant_grays),
            "borderWidth": 1
        }]
    }
    
    # Analyze contrast
    contrast_level, contrast_value = analyze_grayscale_contrast(histogram)
    
    # Determine emotional perception based on tonal distribution
    emotions = determine_monochrome_emotional_impact(dominant_grays, contrast_level)
    
    # Determine type of monochrome image
    monochrome_type = "High key" if sum(percentages[:2]) < sum(percentages[3:]) else "Low key"
    
    # Create more detailed description
    if monochrome_type == "High key":
        description = f"This is a high-key monochrome image with {contrast_level.lower()} contrast. High-key images focus on light tones, creating an airy, subtle atmosphere."
    else:
        description = f"This is a low-key monochrome image with {contrast_level.lower()} contrast. Low-key images emphasize shadows and darker tones, often creating a dramatic or moody atmosphere."
    
    # Create result with additional data for compatibility with color analysis
    result = {
        "dominant_colors": dominant_grays,
        "chart_data": chart_data,
        "color_harmony": {
            "type": "Monochrome",
            "description": description,
            "score": 0.8 + (0.2 * (contrast_value / 255.0))  # High score for monochrome images, considering contrast
        },
        "emotional_impact": emotions,
        "is_monochrome": True,
        "monochrome_type": monochrome_type,
        "contrast_level": contrast_level,
        "contrast_value": contrast_value
    }
    
    return result

def analyze_grayscale_contrast(histogram):
    """Analyzes contrast of grayscale image based on histogram"""
    # Find 5th and 95th percentiles for more reliable contrast assessment
    total = sum(histogram)
    cumsum = 0
    idx_5 = 0
    for i, count in enumerate(histogram):
        cumsum += count
        if cumsum / total >= 0.05:
            idx_5 = i
            break
    
    cumsum = 0
    idx_95 = 255
    for i in range(255, -1, -1):
        cumsum += histogram[i]
        if cumsum / total >= 0.05:
            idx_95 = i
            break
    
    # Calculate difference between percentiles as a measure of contrast
    contrast_diff = idx_95 - idx_5
    
    # Determine contrast level
    if contrast_diff > 200:
        return "Very High", contrast_diff
    elif contrast_diff > 150:
        return "High", contrast_diff
    elif contrast_diff > 100:
        return "Medium", contrast_diff
    elif contrast_diff > 50:
        return "Low", contrast_diff
    else:
        return "Very Low", contrast_diff

def determine_monochrome_emotional_impact(dominant_grays, contrast_level):
    """Determines emotional impact of monochrome image"""
    emotions = ["Monochromatic", "Timeless"]
    
    # Based on tonal distribution
    dark_tones_percentage = sum(item["percentage"] for item in dominant_grays if item.get("is_dark", False))
    
    if dark_tones_percentage > 70:
        emotions.extend(["Dramatic", "Mysterious", "Profound"])
    elif dark_tones_percentage < 30:
        emotions.extend(["Airy", "Ethereal", "Delicate"])
    else:
        emotions.extend(["Balanced", "Nuanced"])
    
    # Based on contrast
    if contrast_level in ["Very High", "High"]:
        emotions.extend(["Bold", "Powerful"])
    elif contrast_level in ["Very Low", "Low"]:
        emotions.extend(["Subtle", "Soft"])
    
    # Unique emotions
    return list(set(emotions))

def determine_optimal_clusters(pixels, max_clusters=8):
    """Determines optimal number of color clusters using the elbow method"""
    # For a smaller subset of pixels for faster processing
    sample_size = min(10000, len(pixels))
    pixel_sample = pixels[np.random.choice(len(pixels), size=sample_size, replace=False)]
    
    inertias = []
    silhouettes = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixel_sample)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score does not work for n_clusters=1
        if n_clusters > 1:
            try:
                silhouette = silhouette_score(pixel_sample, labels)
                silhouettes.append(silhouette)
            except:
                silhouettes.append(0)
    
    # Use the elbow method to determine the optimal number of clusters
    try:
        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
        optimal = kl.elbow
    except:
        # If the elbow method did not work, find the maximum silhouette score
        optimal = np.argmax(silhouettes) + 2
    
    # If it was not possible to determine, use 5 by default
    if optimal is None or optimal < 3:
        optimal = 5
        
    return optimal

def analyze_color_properties(centers):
    """Converts cluster centers (RGB) to HSV."""
    hsv_colors = []
    for center in centers:
        # Check if the center is not empty or invalid
        if isinstance(center, (list, np.ndarray)) and len(center) == 3:
            # Convert to floating point numbers in the range 0-1
            r, g, b = [x / 255.0 for x in center]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append((h, s, v))
        else:
            logger.warning(f"Skipping invalid cluster center: {center}")
            
    if not hsv_colors:
        logger.error("No valid HSV colors could be derived from cluster centers.")
        # In the extreme case, return an empty list to avoid an error above
        return []
        
    return hsv_colors

def determine_color_harmony(hsv_colors):
    """Determines type of color harmony based on HSV values"""
    # Extract hues for analysis
    hues = [h for h, _, _ in hsv_colors]
    
    # Check for monochromatic harmony (same hues)
    if max(hues) - min(hues) < 0.08:
        return {
            "type": "Monochromatic",
            "description": "The image uses variations of a single hue, creating a harmonious and cohesive feel."
        }
    
    # Check for analogous colors (adjacent on the color wheel)
    hue_diffs = [abs(hues[i] - hues[j]) % 1.0 for i in range(len(hues)) for j in range(i+1, len(hues))]
    if all(diff < 0.25 or diff > 0.75 for diff in hue_diffs):
        return {
            "type": "Analogous",
            "description": "The image uses colors that are adjacent to each other on the color wheel, creating a serene and comfortable feel."
        }
    
    # Check for complementary colors (opposite on the color wheel)
    complementary_pairs = 0
    for i in range(len(hues)):
        for j in range(i+1, len(hues)):
            diff = abs(hues[i] - hues[j]) % 1.0
            if 0.45 < diff < 0.55:
                complementary_pairs += 1
    
    if complementary_pairs > 0:
        return {
            "type": "Complementary",
            "description": "The image uses colors from opposite sides of the color wheel, creating a vibrant look with high contrast."
        }
    
    # Check for triadic harmony (colors evenly distributed on the color wheel)
    triad_found = False
    if len(hues) >= 3: # Triad requires at least 3 colors
        for i in range(len(hues)):
            for j in range(i+1, len(hues)):
                for k in range(j+1, len(hues)):
                    # Normalize hue difference to be in [0, 0.5]
                    diff1 = abs(hues[i] - hues[j]) % 1.0
                    diff1 = min(diff1, 1.0 - diff1)
                    diff2 = abs(hues[j] - hues[k]) % 1.0
                    diff2 = min(diff2, 1.0 - diff2)
                    diff3 = abs(hues[k] - hues[i]) % 1.0
                    diff3 = min(diff3, 1.0 - diff3)
                    
                    # Check if the differences are close to 1/3 of the circle (approximately 0.33)
                    if all(0.28 < diff < 0.38 for diff in [diff1, diff2, diff3]):
                        triad_found = True
                        break
                if triad_found: break
            if triad_found: break
    
    if triad_found:
        return {
            "type": "Triadic",
            "description": "The image uses three colors that are evenly spaced around the color wheel, creating a balanced yet vibrant look."
        }
    
    # By default, consider it a complex harmony
    return {
        "type": "Complex",
        "description": "The image uses a complex color scheme with multiple hues that create visual interest and depth."
    }

def determine_emotional_impact(hsv_colors, percentages):
    """Determines emotional impact of colors"""
    if not hsv_colors: # If the list is empty
        return ["Neutral", "Unknown due to lack of color data"]
        
    # Calculate average values taking into account weight (percentage)
    total_percentage = sum(percentages)
    if total_percentage == 0: # Avoid division by zero
        return ["Neutral", "Cannot determine impact (zero percentage)"]
        
    avg_hue = sum(h * p for (h, _, _), p in zip(hsv_colors, percentages)) / total_percentage
    avg_saturation = sum(s * p for (_, s, _), p in zip(hsv_colors, percentages)) / total_percentage
    avg_value = sum(v * p for (_, _, v), p in zip(hsv_colors, percentages)) / total_percentage
    
    emotions = []
    
    # Intensity (saturation)
    if avg_saturation > 0.65:
        emotions.append("Vibrant")
    elif avg_saturation < 0.25:
        emotions.append("Subtle")
    else:
        emotions.append("Balanced")
    
    # Brightness
    if avg_value > 0.7:
        emotions.append("Bright")
    elif avg_value < 0.3:
        emotions.append("Dark")
    else:
        emotions.append("Moderate")
    
    # Analysis of dominant hues for emotional impact (weighted average hue)
    hue = avg_hue % 1.0 # Normalize hue
    
    # Determination of emotion based on hue
    if 0.95 <= hue or hue < 0.05:  # Red
        emotions.append("Passionate")
    elif 0.05 <= hue < 0.11:  # Orange
        emotions.append("Energetic")
    elif 0.11 <= hue < 0.18:  # Yellow
        emotions.append("Cheerful")
    elif 0.18 <= hue < 0.45:  # Green
        emotions.append("Natural")
    elif 0.45 <= hue < 0.57:  # Cyan
        emotions.append("Calm")
    elif 0.57 <= hue < 0.75:  # Blue/Purple
        emotions.append("Serene")
    elif 0.75 <= hue < 0.95:  # Purple/Pink
        emotions.append("Romantic")
    
    # Contrast (difference between max and min brightness of dominant colors)
    values = [v for _, _, v in hsv_colors]
    if values:
        contrast = max(values) - min(values)
        if contrast > 0.6:
            emotions.append("Dramatic")
        elif contrast < 0.3:
            emotions.append("Harmonious")
    
    # Uniqueness of emotions
    return list(set(emotions))

def get_closest_color_name(rgb):
    """Determines the closest color name for an RGB value"""
    r, g, b = rgb
    
    # Dictionary of colors (RGB -> name)
    color_dict = {
        # Basic colors
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (255, 0, 255): "Magenta",
        (0, 255, 255): "Cyan",
        (255, 255, 255): "White",
        (0, 0, 0): "Black",
        
        # Shades of gray
        (128, 128, 128): "Gray",
        (192, 192, 192): "Light Gray",
        (64, 64, 64): "Dark Gray",
        
        # Additional colors
        (128, 0, 0): "Maroon",
        (128, 128, 0): "Olive",
        (0, 128, 0): "Dark Green",
        (128, 0, 128): "Purple",
        (0, 128, 128): "Teal",
        (0, 0, 128): "Navy",
        
        # Pastel tones
        (255, 182, 193): "Light Pink",
        (255, 160, 122): "Light Salmon",
        (250, 250, 210): "Light Yellow",
        (144, 238, 144): "Light Green",
        (173, 216, 230): "Light Blue",
        (221, 160, 221): "Plum",
        
        # Other popular colors
        (165, 42, 42): "Brown",
        (255, 69, 0): "Orange Red",
        (255, 165, 0): "Orange",
        (189, 183, 107): "Khaki",
        (240, 230, 140): "Khaki Light",
        (50, 205, 50): "Lime Green",
        (46, 139, 87): "Sea Green",
        (64, 224, 208): "Turquoise",
        (30, 144, 255): "Dodger Blue",
        (138, 43, 226): "Blue Violet",
        (75, 0, 130): "Indigo",
        (238, 130, 238): "Violet",
        (219, 112, 147): "Pale Violet Red",
        (255, 105, 180): "Hot Pink",
        (188, 143, 143): "Rosy Brown",
        (245, 222, 179): "Wheat",
        (210, 180, 140): "Tan",
        (244, 164, 96): "Sandy Brown",
        (205, 133, 63): "Peru",
        (222, 184, 135): "Burlywood",
        (160, 82, 45): "Sienna"
    }
    
    # Russian color names
    russian_color_names = {
        "Red": "Красный",
        "Green": "Зеленый",
        "Blue": "Синий",
        "Yellow": "Желтый",
        "Magenta": "Пурпурный",
        "Cyan": "Голубой",
        "White": "Белый",
        "Black": "Черный",
        "Gray": "Серый",
        "Light Gray": "Светло-серый",
        "Dark Gray": "Темно-серый",
        "Maroon": "Бордовый",
        "Olive": "Оливковый",
        "Dark Green": "Темно-зеленый",
        "Purple": "Фиолетовый",
        "Teal": "Бирюзовый",
        "Navy": "Темно-синий",
        "Light Pink": "Светло-розовый",
        "Light Salmon": "Светло-лососевый",
        "Light Yellow": "Светло-желтый",
        "Light Green": "Светло-зеленый",
        "Light Blue": "Светло-голубой",
        "Plum": "Сливовый",
        "Brown": "Коричневый",
        "Orange Red": "Оранжево-красный",
        "Orange": "Оранжевый",
        "Khaki": "Хаки",
        "Khaki Light": "Светлый хаки",
        "Lime Green": "Лаймовый",
        "Sea Green": "Морской зеленый",
        "Turquoise": "Бирюзовый",
        "Dodger Blue": "Ярко-голубой",
        "Blue Violet": "Сине-фиолетовый",
        "Indigo": "Индиго",
        "Violet": "Фиолетовый",
        "Pale Violet Red": "Бледно-фиолетово-красный",
        "Hot Pink": "Ярко-розовый",
        "Rosy Brown": "Розово-коричневый",
        "Wheat": "Пшеничный",
        "Tan": "Загар",
        "Sandy Brown": "Песочно-коричневый",
        "Peru": "Перу",
        "Burlywood": "Светло-коричневый",
        "Sienna": "Сиена"
    }
    
    min_distance = float('inf')
    closest_color = "Unknown"
    
    # Find the closest color in our dictionary
    for (r2, g2, b2), name in color_dict.items():
        # Distance between colors (Euclidean)
        distance = ((r - r2) ** 2 + (g - g2) ** 2 + (b - b2) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    # Return the Russian name if available
    return f"{closest_color} ({russian_color_names.get(closest_color, 'Unknown')})"

def legacy_color_analysis(img):
    """Backup function for color analysis when improved libraries are not available"""
    # Simple implementation of color analysis
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Reduce image size for faster analysis
    img_small = img.resize((100, 100))
    
    # Extract all pixels
    pixels = list(img_small.getdata())
    
    # Count the number of each color
    color_count = {}
    for pixel in pixels:
        # Quantize colors to reduce variations
        quantized = (pixel[0] // 32 * 32, pixel[1] // 32 * 32, pixel[2] // 32 * 32)
        if quantized in color_count:
            color_count[quantized] += 1
        else:
            color_count[quantized] = 1
    
    # Sort by frequency
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top 5 colors
    top_colors = sorted_colors[:5]
    
    # Prepare the result
    dominant_colors = []
    labels = []
    hex_colors = []
    dataset = []
    
    total_pixels = len(pixels)
    for i, ((r, g, b), count) in enumerate(top_colors):
        percentage = (count / total_pixels) * 100
        hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        # Simple color name
        if r > g and r > b:
            name = "Red Shade"
        elif g > r and g > b:
            name = "Green Shade"
        elif b > r and b > g:
            name = "Blue Shade"
        elif r > 200 and g > 200 and b > 200:
            name = "White Shade"
        elif r < 50 and g < 50 and b < 50:
            name = "Black Shade"
        else:
            name = "Mixed Shade"
        
        name = f"Color {i+1}: {name}"
        
        dominant_colors.append({
            "rgb": [r, g, b],
            "hex": hex_code,
            "percentage": round(percentage, 1),
            "name": name
        })
        
        labels.append(name)
        hex_colors.append(hex_code)
        dataset.append(round(percentage, 1))
    
    # Create data for the chart
    chart_data = {
        "labels": labels,
        "datasets": [{
            "data": dataset,
            "backgroundColor": hex_colors,
            "borderColor": ["#ffffff"] * len(hex_colors),
            "borderWidth": 1
        }]
    }
    
    # Form the result
    result = {
        "dominant_colors": dominant_colors,
        "chart_data": chart_data,
        "color_harmony": {
            "type": "Basic",
            "description": "Color harmony analysis not available in legacy mode."
        },
        "emotional_impact": ["Basic analysis", "Limited details"],
        "is_monochrome": False,
        "version": "legacy"
    }
    
    return result

