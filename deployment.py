from FractureDetector import YoloDetector
from BodyPartClassifer import EfficientNetClassifier
from Interface import GradioApp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')


if __name__ == "__main__":
    classifier = EfficientNetClassifier(model_path='SavedModel/efficientnet_body_part_model.pth')
    fracture_detector = YoloDetector(model_path='SavedModel/Yolo_fracture_detection_model.pt')
    app = GradioApp(classifier, fracture_detector)
    app.launch()