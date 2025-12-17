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
    "image_name",
    "worm_id",
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
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape

    # ---------- MASK MICROSCOPE ----------
    mask = np.zeros_like(gray)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 10, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # ---------- ENHANCE WORMS ----------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(masked, cv2.MORPH_BLACKHAT, kernel)

    # ---------- THRESHOLD ----------
    thresh = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5
    )
    thresh = cv2.bitwise_not(thresh)

    # ---------- CLEAN ----------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # ---------- FIND CONTOURS ----------
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    min_area = 0.0005 * (h * w)
    worms = [c for c in contours if cv2.contourArea(c) > min_area]

    print(f"{filename}: detected {len(worms)} worms")

    # ---------- EXTRACT COORDINATES ----------
    for worm_id, worm in enumerate(worms, start=1):
        pts = worm.reshape(-1, 2)

        left   = tuple(pts[pts[:, 0].argmin()])
        right  = tuple(pts[pts[:, 0].argmax()])
        top    = tuple(pts[pts[:, 1].argmin()])
        bottom = tuple(pts[pts[:, 1].argmax()])

        writer.writerow([
            filename,
            worm_id,
            left[0], left[1],
            right[0], right[1],
            top[0], top[1],
            bottom[0], bottom[1]
        ])

        for p in [left, right, top, bottom]:
            cv2.circle(img, p, 4, (0, 0, 255), -1)

    # ---------- SAVE ANNOTATED IMAGE ----------
    output_image_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_image_path, img)

# =========================
# CLEANUP
# =========================
csv_file.close()
print("Batch processing complete.")
