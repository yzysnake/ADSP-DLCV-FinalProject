import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import pydicom
from PIL import Image
import numpy as np
import os


class EfficientNetClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.body_dict = {
            'abdomen': 0,
            'ankle': 1,
            'cervical spine': 2,
            'chest': 3,
            'clavicles': 4,
            'elbow': 5,
            'feet': 6,
            'finger': 7,
            'forearm': 8,
            'hand': 9,
            'hip': 10,
            'knee': 11,
            'leg': 12,
            'lumbar spine': 13,
            'others': 14,
            'pelvis': 15,
            'shoulder': 16,
            'sinus': 17,
            'skull': 18,
            'thigh': 19,
            'thoracic spine': 20,
            'wrist': 21,
            'mixed': 22
        }
        self.label_to_body_part = {v: k for k, v in self.body_dict.items()}
        self.num_classes = len(self.body_dict)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = models.efficientnet_b5(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def convert_dcm_to_jpg(self, dcm_path, jpg_path):
        try:
            dicom = pydicom.dcmread(dcm_path)
            img = dicom.pixel_array
            img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(jpg_path)
            return jpg_path
        except Exception as e:
            print(f"Error converting {dcm_path} to JPG: {e}")
            return None

    def predict_image(self, image_path):
        if image_path.endswith('.dcm'):
            jpg_path = image_path.replace('.dcm', '.jpg')
            if not os.path.exists(jpg_path):
                jpg_path = self.convert_dcm_to_jpg(image_path, jpg_path)
            image_path = jpg_path

        image = Image.open(image_path).convert("RGB")
        transformed_image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(transformed_image)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()

        class_name = self.label_to_body_part.get(predicted_class, "Unknown")
        return class_name
