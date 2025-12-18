# Worm Coordinate Detector 🪱🔬

A Python-based Computer Vision tool designed to detect biological worms in microscope images and extract their endpoint coordinates. This tool was developed to generate high-accuracy training data for Machine Learning models.

## 📌 Project Overview

This project solves the challenge of extracting precise coordinates of worms from noisy microscope slides. It allows for:
1.  **Automatic Detection:** Using Computer Vision (HSV Color + Morphology) to detect blue/purple worms.
2.  **Semi-Automatic Detection:** Detecting manually marked "Red Dots" for 100% accuracy in difficult images.
3.  **Data formatting:** Scaling all coordinates to a standardized 256x256 grid for ML training.

## 🚀 Features

* **Dual Detection Modes:**
    * *Auto-Mode:* Detects thick blue/purple lines using Saturation and Green-channel analysis.
    * *Manual-Mode:* Detects red dots drawn by humans and pairs them into "Head & Tail" coordinates.
* **Coordinate Scaling:** Automatically normalizes coordinates from original image resolution to a target size (e.g., 256x256).
* **Confidence Scoring:** (Auto-mode) Calculates a confidence percentage based on shape, aspect ratio, and solidity.
* **Visual Verification:** Generates annotated images with bounding boxes, center lines, and endpoints to verify accuracy.
* **Batch Processing:** Processes entire folders of images at once.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/worm-detector.git](https://github.com/yourusername/worm-detector.git)
    cd worm-detector
    ```

2.  **Install Dependencies:**
    This project requires Python 3.x and the following libraries:
    ```bash
    pip install opencv-python numpy
    ```

## 📂 Folder Structure
worm-detector/ ├── images/ # Put your raw microscope images here (.jpg, .png) ├── output/ │ ├── annotated/ # Output images with drawn boxes/lines will be saved here │ └── worm_coordinates.csv # The final CSV data file ├── detect_manual_dots.py # Script for Method A (Red Dot Markers) ├── worm_detect.py # Script for Method B (Automatic Detection) └── README.md

## ⚙️ Usage

### Method A: Manual Markers (Recommended for Accuracy)
Use this method if the microscope images are too noisy for automatic detection.

1.  Open your images in any editor (Paint, Photoshop, etc.).
2.  Draw a **Red Dot** on the head and tail of every worm.
3.  Save the images in the `images/` folder.
4.  Run the script:
    ```bash
    python detect_manual_dots.py
    ```
5.  Check `output/annotated/` to see green lines connecting your dots.

### Method B: Automatic Detection
Use this method for clean slides with high-contrast blue/purple worms.

1.  Place raw images in the `images/` folder.
2.  Run the script:
    ```bash
    python worm_detect.py
    ```
3.  The script will filter noise, detect worm shapes based on aspect ratio (> 3.5), and extract endpoints.

## 📊 Output Format

The results are saved in `output/worm_coordinates.csv`. The coordinates are scaled to **0-256**.

| image_name | worm_id | end1_x | end1_y | end2_x | end2_y | confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| slide_1.jpg | 1 | 34 | 50 | 88 | 120 | 95% |
| slide_1.jpg | 2 | 150 | 200 | 180 | 220 | 88% |

* **end1_x / y**: Coordinates of the first endpoint.
* **end2_x / y**: Coordinates of the second endpoint.
* **confidence**: (Auto-mode only) How likely the object is a worm.

## 🧠 Technical Details

* **Color Space:** Utilizes HSV for color masking (Red dots) and Green-channel extraction for purple worms.
* **Morphology:** Uses `MORPH_OPEN` and `DILATE` operations to remove dust noise and connect broken worm segments.
* **Geometric Filtering:** Filters false positives by calculating **Circularity** and **Aspect Ratio** (Worms must be elongated, Dots are circular).

## 📝 License

This project is open-source and available for educational and research purposes.
