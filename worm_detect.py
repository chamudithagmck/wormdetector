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

    original = img.copy()

    # ---------- PREPROCESS ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Mask the Microscope Circle (Remove Black Borders)
    h, w = gray.shape
    mask = np.zeros_like(gray)
    # Draw a white circle in the center (slightly smaller than image) to define the "safe zone"
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 30, 255, -1)
    
    # Apply mask: Everything outside the circle becomes black
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # ---------- ENHANCE WORMS (Blackhat) ----------
    # Blackhat extracts "Dark objects on Light background"
    # The worms are dark purple on light gray.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(masked_gray, cv2.MORPH_BLACKHAT, kernel)

    # ---------- THRESHOLD ----------
    # This makes the "enhanced" worms White and background Black
    thresh = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        31,
        -5
    )

    # *** REMOVED bitwise_not HERE ***
    # We want White Worms on Black Background for findContours.

    # ---------- CLEAN NOISE ----------
    # Remove small white dots (purple stain noise)
    # MORPH_OPEN removes small white noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    # Dilate slightly to connect broken parts of the worm
    clean = cv2.dilate(clean, kernel_clean, iterations=2)

    # ---------- FIND CONTOURS ----------
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by Area (Adjust this based on worm size)
    # If worms are not detected, lower this number
    min_area = 500 
    worms = [c for c in contours if cv2.contourArea(c) > min_area]

    print(f"{filename}: detected {len(worms)} worms")

    # ---------- EXTRACT COORDINATES ----------
    for worm_id, worm in enumerate(worms, start=1):
        # 4 Extreme Points
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

        # Draw green box
        x, y, w_box, h_box = cv2.boundingRect(worm)
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # Draw red dots
        for p in [extLeft, extRight, extTop, extBot]:
            cv2.circle(img, p, 5, (0, 0, 255), -1)

    # ---------- SAVE ANNOTATED IMAGE ----------
    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, img)

# =========================
# CLEANUP
# =========================
csv_file.close()
print("Batch processing complete.")