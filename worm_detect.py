import cv2
import numpy as np
import csv
import os

# =========================
# PATHS
# =========================
IMAGE_DIR = "images"
OUTPUT_DIR = "output/annotated"
CSV_PATH = "output/worm_coordinates.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CSV SETUP
# =========================
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "image_name", "worm_id",
    "left_x", "left_y",
    "right_x", "right_y",
    "top_x", "top_y",
    "bottom_x", "bottom_y"
])

# =========================
# PROCESS EACH IMAGE
# =========================
for filename in os.listdir(IMAGE_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping {filename} (cannot read)")
        continue

    # ---------- PREPROCESS ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Mask the Microscope Circle
    mask = np.zeros_like(gray)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 50, 255, -1)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # 2. Strong Contrast Enhancement (CLAHE)
    # This helps faint worms stand out against the gray background
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)

    # 3. Adaptive Thresholding (More Sensitive)
    # Block Size 21 (smaller than before) to catch thinner features
    # C = 5 (higher constant) to reduce background noise
    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, # Inverted: Worms become WHITE, Background BLACK
        21,
        5
    )

    # 4. Morphological Closing (Connect broken pieces)
    # This connects parts of a worm if they are slightly separated
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect, iterations=2)

    # 5. Remove Small Noise (Opening)
    # Removes tiny dots without removing thin worms
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # ---------- FIND CONTOURS ----------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_worms = []
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # FILTER 1: AREA
        # Lowered to 100 to catch small/broken worm pieces
        if area < 100: 
            continue
            
        # FILTER 2: ELONGATION (The most important filter!)
        # Worms are long. Dots are round.
        if len(c) < 5: # Need enough points to fit an ellipse
            continue
            
        # Fit an ellipse to get length and width
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        
        # Avoid division by zero
        if MA == 0: continue
            
        # Aspect Ratio (Major Axis / Minor Axis)
        # Round dots have ratio close to 1. Worms have high ratio.
        aspect_ratio = ma / MA
        
        if aspect_ratio < 2.5: # If it's too round, it's a stain/dot. Skip it.
            continue
            
        valid_worms.append(c)

    # Sort largest to smallest and take top 15 (prevent noise explosion)
    valid_worms = sorted(valid_worms, key=cv2.contourArea, reverse=True)[:15]

    print(f"{filename}: detected {len(valid_worms)} worms")

    # ---------- EXTRACT COORDINATES ----------
    for worm_id, worm in enumerate(valid_worms, start=1):
        extLeft = tuple(worm[worm[:, :, 0].argmin()][0])
        extRight = tuple(worm[worm[:, :, 0].argmax()][0])
        extTop = tuple(worm[worm[:, :, 1].argmin()][0])
        extBot = tuple(worm[worm[:, :, 1].argmax()][0])

        writer.writerow([
            filename,
            worm_id,
            extLeft[0], extLeft[1],
            extRight[0], extRight[1],
            extTop[0], extTop[1],
            extBot[0], extBot[1]
        ])

        # Visualization
        x, y, w_box, h_box = cv2.boundingRect(worm)
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        for p in [extLeft, extRight, extTop, extBot]:
            cv2.circle(img, p, 3, (0, 0, 255), -1)

    # ---------- SAVE ANNOTATED IMAGE ----------
    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, img)

csv_file.close()
print("Batch processing complete.")