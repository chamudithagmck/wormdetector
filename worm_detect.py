import cv2
import numpy as np
import csv
import os

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
# CSV Header: ONLY Endpoints and Confidence (No Box Coordinates)
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
# HELPER: Calculate Confidence
# =========================
def calculate_confidence(contour, aspect_ratio, area):
    shape_score = min(100, (aspect_ratio / 3.0) * 100)
    area_score = min(100, (area / 1500.0) * 100)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity_score = 0
    if hull_area > 0:
        solidity_score = (float(cv2.contourArea(contour)) / hull_area) * 100
        
    confidence = (shape_score * 0.5) + (area_score * 0.3) + (solidity_score * 0.2)
    return int(min(99, confidence))

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

    # --- 1. DETECTION (Saturation + Green Channel) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    green_channel = img[:, :, 1]
    inverted_green = 255 - green_channel
    
    combined_feature = cv2.addWeighted(saturation, 0.6, inverted_green, 0.4, 0)

    mask = np.zeros_like(combined_feature)
    cv2.circle(mask, (orig_w // 2, orig_h // 2), min(orig_w, orig_h) // 2 - 40, 255, -1)
    masked_feature = cv2.bitwise_and(combined_feature, combined_feature, mask=mask)

    blurred = cv2.GaussianBlur(masked_feature, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.dilate(clean, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_worms = []
    for c in contours:
        area = cv2.contourArea(c)
        
        # STRICT FILTERS (Removes Dots)
        if area < 500: continue
        
        if len(c) < 5: continue
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        if MA == 0: continue
        
        aspect_ratio = ma / MA
        if aspect_ratio < 1.8: continue 
        
        conf = calculate_confidence(c, aspect_ratio, area)
        if conf < 40: continue

        valid_worms.append((c, conf))

    # Keep top 10 strongest detections
    valid_worms = sorted(valid_worms, key=lambda x: x[1], reverse=True)[:10]
    
    print(f"{filename}: detected {len(valid_worms)} worms")

    # --- 2. OUTPUT & VISUALIZATION ---
    resized_img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    for worm_id, (worm, confidence) in enumerate(valid_worms, start=1):
        # A. ENDPOINTS
        end1, end2 = get_worm_endpoints(worm)
        s_end1 = (scale_val(end1[0], orig_w, TARGET_SIZE), scale_val(end1[1], orig_h, TARGET_SIZE))
        s_end2 = (scale_val(end2[0], orig_w, TARGET_SIZE), scale_val(end2[1], orig_h, TARGET_SIZE))

        # B. WRITE TO CSV (Clean: No Box Coords)
        writer.writerow([
            filename, worm_id, f"{confidence}%",
            s_end1[0], s_end1[1],
            s_end2[0], s_end2[1]
        ])

        # C. DRAW ON IMAGE (Includes Box for Verification)
        x, y, w, h = cv2.boundingRect(worm)
        s_x = scale_val(x, orig_w, TARGET_SIZE)
        s_y = scale_val(y, orig_h, TARGET_SIZE)
        s_w = scale_val(w, orig_w, TARGET_SIZE)
        s_h = scale_val(h, orig_h, TARGET_SIZE)

        # Green Box
        cv2.rectangle(resized_img, (s_x, s_y), (s_x + s_w, s_y + s_h), (0, 255, 0), 1)
        
        # Yellow Line
        cv2.line(resized_img, s_end1, s_end2, (0, 255, 255), 1)
        
        # Red Endpoints
        cv2.circle(resized_img, s_end1, 3, (0, 0, 255), -1)
        cv2.circle(resized_img, s_end2, 3, (0, 0, 255), -1)
        
        # Confidence Text
        cv2.putText(resized_img, f"{confidence}%", (s_x, s_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, resized_img)

csv_file.close()
print("Done.")