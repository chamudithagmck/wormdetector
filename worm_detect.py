import cv2
import numpy as np
import csv
import os
import math

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
# Now we only save 2 coordinates: Start (x1,y1) and End (x2,y2)
writer.writerow(["image_name", "worm_id", "end1_x", "end1_y", "end2_x", "end2_y"])

# =========================
# HELPER: Get 2 Endpoints
# =========================
def get_worm_endpoints(contour):
    # Method: We find the two points in the contour that are farthest apart.
    # This is computationally expensive for huge shapes, but perfect for small worms.
    
    # Reshape contour to simple list of points
    pts = contour.reshape(-1, 2)
    
    max_dist = 0
    p1, p2 = (0,0), (0,0)
    
    # To speed it up, we only check points on the Convex Hull (outer edge)
    hull = cv2.convexHull(contour)
    hull_pts = hull.reshape(-1, 2)
    
    # Brute-force distance check on hull points (reliable for finding ends)
    for i in range(len(hull_pts)):
        for j in range(i + 1, len(hull_pts)):
            dist = np.linalg.norm(hull_pts[i] - hull_pts[j])
            if dist > max_dist:
                max_dist = dist
                p1 = tuple(hull_pts[i])
                p2 = tuple(hull_pts[j])
                
    return p1, p2

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

    # --- DETECTION LOGIC (Same as before) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (orig_w // 2, orig_h // 2), min(orig_w, orig_h) // 2 - 40, 255, -1)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)

    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 5)

    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_worms = []
    for c in contours:
        if cv2.contourArea(c) < 100: continue
        if len(c) < 5: continue
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        if MA == 0: continue
        if (ma / MA) < 2.0: continue
        valid_worms.append(c)

    valid_worms = sorted(valid_worms, key=cv2.contourArea, reverse=True)[:15]
    print(f"{filename}: detected {len(valid_worms)} worms")

    # --- EXTRACT 2 ENDPOINTS ---
    resized_img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    def scale_val(val, original_max, target_max):
        return int((val / original_max) * target_max)

    for worm_id, worm in enumerate(valid_worms, start=1):
        # 1. Get the two farthest points on the worm (Head & Tail)
        end1, end2 = get_worm_endpoints(worm)
        
        # 2. Scale them to 256x256
        s_end1 = (scale_val(end1[0], orig_w, TARGET_SIZE), scale_val(end1[1], orig_h, TARGET_SIZE))
        s_end2 = (scale_val(end2[0], orig_w, TARGET_SIZE), scale_val(end2[1], orig_h, TARGET_SIZE))

        # 3. Save to CSV
        writer.writerow([
            filename, worm_id,
            s_end1[0], s_end1[1],
            s_end2[0], s_end2[1]
        ])

        # 4. Draw Visuals
        # Draw a line connecting the two ends (Visual check)
        cv2.line(resized_img, s_end1, s_end2, (0, 255, 255), 1) 
        # Draw Blue dots at the ends
        cv2.circle(resized_img, s_end1, 3, (255, 0, 0), -1)
        cv2.circle(resized_img, s_end2, 3, (255, 0, 0), -1)

    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, resized_img)

csv_file.close()
print("Done.")