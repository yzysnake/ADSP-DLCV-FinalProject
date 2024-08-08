import gradio as gr


class GradioApp:
    def __init__(self, classifier, fracture_detector):
        self.classifier = classifier
        self.fracture_detector = fracture_detector

    def classify_image(self, image_path):
        class_name = self.classifier.predict_image(image_path)
        image_with_boxes, num_boxes = self.fracture_detector.predict_boxes(image_path)
        fracture_message = f"{num_boxes} fracture(s) detected" if num_boxes > 0 else "No fractures detected"
        return class_name, image_with_boxes, fracture_message

    def launch(self):
        iface = gr.Interface(
            fn=self.classify_image,
            inputs=gr.Image(type="filepath", label="Upload an Image"),
            outputs=[
                gr.Textbox(label="Body Part", lines=2, placeholder="Predicted body part class will appear here"),
                gr.Image(label="Fracture Detection"),
                gr.Textbox(label="Fracture Detection Message", lines=2,
                           placeholder="Fracture detection result will appear here")
            ],
            title="X-ray Body Part Classifier & Fracture Detection",
            description="Upload an X-ray image (jpg or DICOM) to classify the body part and detect fractures.",
            theme=gr.themes.Monochrome(primary_hue='red')
        )
        iface.launch(share=True)