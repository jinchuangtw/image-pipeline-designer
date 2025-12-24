# image-pipeline-designer

A lightweight image-processing pipeline design tool built with **PyQt5 + OpenCV** for **Python 3.8.20**. This tool allows users to interactively design, preview, and apply traditional image-processing pipelines with safe parameter controls, undo/redo support, and a readable history log.

[TOC]

## Features

#### 1. Image Preview

- **Before**: the current committed image in the pipeline
- **After**: live preview of the selected operation applied to the current image

#### 2. Image Operations

All parameters are adjusted via sliders or selectors to prevent invalid input.

Supported operations:

- Average Blur
- Gaussian Blur
- Median Blur
- Threshold (non-adaptive)
- Adaptive Threshold
- Otsu Threshold
- Sobel (X/Y/Magnitude)
- Canny
- Morphology (Erode, Dilate, Open, Close, Gradient, TopHat, BlackHat)
- Difference (with a user-selected second image)
- CLAHE
- Unsharp Mask
- Gamma Correction

#### 3. Pipeline Control

- **Apply Step**: commit the current preview as a pipeline stage
- **Undo / Redo**: navigate pipeline history
- **Reset**: return to the original image

#### 4. History Log

A text-based history records all applied steps and parameters in order, including reference image names for difference operations.

## Requirements

- Python **3.8.20**
- PyQt5
- OpenCV (opencv-python)
- NumPy

## Recommended Setup (conda)

#### 1. Create Environment

```bash
conda create -n img-pipeline python=3.8.20
conda activate img-pipeline
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

From the project directory:

```bash
python main.py
```

## Basic Usage

1. Click `Load Image` to open an image.
2. Select an operation from the dropdown menu.
3. Adjust parameters using the provided controls.
4. Observe the live result in the `After` panel.
5. Click `Apply Step` to add the operation to the pipeline.
6. Use `Undo` / `Redo` to navigate history, or `Reset` to start over.
