import torch
import pydicom
from PIL import Image
import numpy as np
import cv2


class YoloDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)

    def preprocess_image(self, image_path, target_size=(640, 640)):
        if image_path.endswith('.dcm'):
            jpg_path = self.convert_dcm_to_jpg(image_path)
            image = cv2.imread(jpg_path)
        else:
            image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Original size
        original_size = image.shape[:2]  # (height, width)

        # Resize image
        image_resized = cv2.resize(image, target_size)

        return image_resized, original_size

    def convert_dcm_to_jpg(self, dcm_path):
        dicom = pydicom.dcmread(dcm_path)
        img = dicom.pixel_array

        # Normalize to 8-bit range
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        jpg_path = dcm_path.replace('.dcm', '.jpg')
        Image.fromarray(img).save(jpg_path)

        return jpg_path

    def draw_boxes(self, image, boxes):
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        return image

    def predict_boxes(self, image_path):
        image_resized, original_size = self.preprocess_image(image_path)
        results = self.model(image_resized)

        # YOLOv5 results are already in the format [x1, y1, x2, y2, conf, class]
        predicted_boxes = results.xyxy[0].cpu().numpy()

        image_with_boxes = self.draw_boxes(image_resized.copy(), predicted_boxes)
        num_boxes = len(predicted_boxes)

        return image_with_boxes, num_boxes
