import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Klassennamen
class_names = ['Defect', 'Filled', 'Good']

# Predictor Klasse
class Predictor:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            return self.model(image_tensor)

# Konfigurieren Sie die Transformationskette für das Modell
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(img_np_array, predictor):
    # Konvertieren Sie das NumPy-Array in ein PIL-Bild und wenden Sie die Transformation an
    img = Image.fromarray(img_np_array.astype('uint8'), 'RGB')
    img_tensor = transform(img).unsqueeze(0) 
    print(img_tensor.dtype) 

    # Führen Sie die Vorhersage mit dem Modell durch
    output_logits = predictor.predict(img_tensor)
    probabilities = torch.softmax(output_logits, dim=1)
    pred_label = torch.argmax(probabilities, dim=1).item()
    pred_class = class_names[pred_label]
    probability = probabilities.max()*100

    return pred_class, probability