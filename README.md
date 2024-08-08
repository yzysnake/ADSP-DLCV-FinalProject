# X-Ray Image Analysis for Body Part Classifier & Bone Fracture Detection

## Project Description

Reading X-ray images is a crucial step for patients' treatment, but it is also very time-consuming for radiologists. Bone fractures commonly go undetected, leading to complications and delays in patients' care. The increased interest in computer-aided diagnosis can reduce radiologists' burden and improve their detection of bone fractures.

This project addresses two primary problems:

### Problem 1: Image Classification
Prediction of the specific body part pictured in an X-ray.

- **Classes**: 22 classes focused on cases where there is one body part - others termed "mixed".
- **Models**: Custom CNN and EfficientNet.

### Problem 2: Object Detection
Combination of classification and regression problem, focused on identifying the presence and location of fractures in X-rays.

- **Models**: Faster R-CNN and YOLO.

## Datasets
We leverage two datasets for this project:

1. [FracAtlas](https://datasetninja.com/frac-atlas#introduction)
2. [UNIFESP X-ray Bodypart Classification](https://www.kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification/data?select=t
rain.csv)

Please download and unzip these datasets into the `data` directory.

## Models Used
- **EfficientNet**: For image classification.
- **YOLO**: For object detection.

## Demonstration
We use Gradio to deploy the models as a web-based example.

To use it, run deployment.py

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt

