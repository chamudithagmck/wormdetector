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
writer.writerow([
    "image_name", "worm_id", 
    "end1_x", "end1_y", 
    "end2_x", "end2_y"
])

# =========================
# HELPER: Scale Coordinates
# =========================
def scale_val(val, original_max, target_max):
    return int((val / original_max) * target_max)

# =========================
# HELPER: Group Points into Pairs
# =========================
def group_dots_into_worms(points):
    # If we have an odd number of dots, one will be left alone
    pairs = []
    used = [False] * len(points)
    
    for i in range(len(points)):
        if used[i]: continue
        
        # Find the closest unused point to point[i]
        min_dist = float('inf')
        match_index = -1
        
        for j in range(i + 1, len(points)):
            if used[j]: continue
            
            # Euclidean distance
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            if dist < min_dist:
                min_dist = dist
                match_index = j
        
        # If we found a pair
        if match_index != -1:
            pairs.append((points[i], points[match_index]))
            used[i] = True
            used[match_index] = True
            
    return pairs

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

    # 1. Convert to HSV (Best for color detection)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Define "Blue" Color Range (Tuned for #0078d7)
    # Target is Hue=103, Saturation=255, Value=215.
    # We set a range of +/- 10 on Hue to catch slight variations.
    
    lower_blue = np.array([93, 150, 150])  # Min Hue 93, High Saturation
    upper_blue = np.array([113, 255, 255]) # Max Hue 113

    # Create Mask (Only one mask needed for Blue)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean noise (remove single pixel specks)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Find Contours (The Dots)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_points = []
    for c in contours:
        # Filter tiny noise (must be at least 5 pixels)
        if cv2.contourArea(c) < 5: continue
        
        # Get center of the dot
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_points.append((cx, cy))

    # 4. Group Dots into Worms (Pairs)
    worm_pairs = group_dots_into_worms(detected_points)
    
    print(f"{filename}: Found {len(detected_points)} dots -> {len(worm_pairs)} worms")

    # 5. Output
    resized_img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    for worm_id, (p1, p2) in enumerate(worm_pairs, start=1):
        # Scale to 256x256
        s_p1 = (scale_val(p1[0], orig_w, TARGET_SIZE), scale_val(p1[1], orig_h, TARGET_SIZE))
        s_p2 = (scale_val(p2[0], orig_w, TARGET_SIZE), scale_val(p2[1], orig_h, TARGET_SIZE))

        # Save to CSV
        writer.writerow([
            filename, worm_id,
            s_p1[0], s_p1[1],
            s_p2[0], s_p2[1]
        ])

        # Draw Verification
        # Yellow Line connecting the dots
        cv2.line(resized_img, s_p1, s_p2, (0, 255, 255), 1)
        # Blue Circles around your Markers (to confirm detection)
        cv2.circle(resized_img, s_p1, 3, (255, 0, 0), 1)
        cv2.circle(resized_img, s_p2, 3, (255, 0, 0), 1)

    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, resized_img)

csv_file.close()
print("Manual Blue Dot extraction complete.")