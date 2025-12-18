import cv2
import numpy as np
import csv
import os
from skimage.filters import frangi, threshold_otsu
from skimage import exposure

# =========================
# CONFIGURATION
# =========================
IMAGE_DIR = "images"
OUTPUT_DIR = "output/annotated"
CSV_PATH = "output/worm_coordinates.csv"
TARGET_SIZE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CSV SETUP
# =========================
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "image_name", "worm_id", "confidence",
    "end1_x", "end1_y", "end2_x", "end2_y"
])

# =========================
# HELPER: Get 2 Endpoints
# =========================
def get_worm_endpoints(contour):
    hull = cv2.convexHull(contour)
    hull_pts = hull.reshape(-1, 2)
    max_dist = 0
    p1, p2 = (0,0), (0,0)
    
    # Optimization
    if len(hull_pts) > 50: hull_pts = hull_pts[::2]

    for i in range(len(hull_pts)):
        for j in range(i + 1, len(hull_pts)):
            dist = np.linalg.norm(hull_pts[i] - hull_pts[j])
            if dist > max_dist:
                max_dist = dist
                p1 = tuple(hull_pts[i])
                p2 = tuple(hull_pts[j])
    return p1, p2

# =========================
# HELPER: Scale Coordinates
# =========================
def scale_val(val, original_max, target_max):
    return int((val / original_max) * target_max)

# =========================
# PROCESS EACH IMAGE
# =========================
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(image_path)
    if img is None: continue

    orig_h, orig_w = img.shape[:2]
    
    # 1. Convert to Green Channel (Best for purple worms)
    # If worms are faint, Green channel usually captures them best as "dark" objects
    green = img[:, :, 1]
    
    # Invert so worms are light and background is dark
    gray = cv2.bitwise_not(green)

    # 2. FRANGI VESSELNESS FILTER (The Core Magic)
    # This filter looks for "tubular" structures.
    # sigmas: range of worm thicknesses to look for (1 to 3 pixels wide)
    # black_ridges=False: We are looking for bright lines (since we inverted the image)
    vesselness = frangi(gray, sigmas=range(1, 4), black_ridges=False)

    # 3. Enhance the result
    # Frangi output is very faint (0.0 to 0.0001), so we rescale it to 0-255
    vesselness = exposure.rescale_intensity(vesselness, out_range=(0, 255)).astype(np.uint8)

    # 4. Thresholding the Vesselness Map
    # Now we only threshold the "Tubeness", not the raw image.
    # This removes 99% of the dots automatically.
    thresh_val = threshold_otsu(vesselness)
    binary = vesselness > (thresh_val * 0.5) # Lower threshold slightly to keep faint worms
    binary = (binary * 255).astype(np.uint8)

    # 5. Clean Up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.dilate(clean, kernel, iterations=2) # Re-connect broken worms

    # 6. Find Contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_worms = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200: continue # Small fragments
        
        # We can be loose with Aspect Ratio now because Frangi already killed the dots
        if len(c) < 5: continue
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        if MA == 0: continue
        
        aspect_ratio = ma / MA
        if aspect_ratio < 2.0: continue # Still filter perfect circles

        # Confidence is high because Frangi already filtered the shape
        conf = min(99, int(area / 1000 * 100))
        valid_worms.append((c, conf))

    # Keep top 15
    valid_worms = sorted(valid_worms, key=lambda x: x[1], reverse=True)[:15]
    print(f"{filename}: detected {len(valid_worms)} worms")

    # --- OUTPUT ---
    resized_img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    for worm_id, (worm, confidence) in enumerate(valid_worms, start=1):
        end1, end2 = get_worm_endpoints(worm)
        s_end1 = (scale_val(end1[0], orig_w, TARGET_SIZE), scale_val(end1[1], orig_h, TARGET_SIZE))
        s_end2 = (scale_val(end2[0], orig_w, TARGET_SIZE), scale_val(end2[1], orig_h, TARGET_SIZE))

        writer.writerow([
            filename, worm_id, f"{confidence}%",
            s_end1[0], s_end1[1],
            s_end2[0], s_end2[1]
        ])

        # Draw Visuals
        x, y, w, h = cv2.boundingRect(worm)
        s_x = scale_val(x, orig_w, TARGET_SIZE)
        s_y = scale_val(y, orig_h, TARGET_SIZE)
        s_w = scale_val(w, orig_w, TARGET_SIZE)
        s_h = scale_val(h, orig_h, TARGET_SIZE)
        
        cv2.rectangle(resized_img, (s_x, s_y), (s_x+s_w, s_y+s_h), (0,255,0), 1)
        cv2.line(resized_img, s_end1, s_end2, (0, 255, 255), 1)
        cv2.circle(resized_img, s_end1, 3, (0, 0, 255), -1)
        cv2.circle(resized_img, s_end2, 3, (0, 0, 255), -1)

    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, resized_img)

csv_file.close()
print("Done.")