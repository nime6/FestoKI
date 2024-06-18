import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision
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
    #transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image_multiclass(img_path, predictor):
    # Konvertieren Sie das NumPy-Array in ein PIL-Bild und wenden Sie die Transformation an
    #print("Hier ist die img array", img_np_array.shape)
    #print("Hier ist die img array", img_np_array)
    #img = Image.fromarray(img_np_array.astype('uint8'), 'RGB')
    #img_tensor = transform(img).unsqueeze(0) 
    
    #print(img_tensor)
    
    #im = Image.fromarray(img.squeeze().permute(1,2,0).numpy())
    #im.save(r"/home/pi/Desktop/FestoKI/FestoKI/test_images" + "test_image.jpg")

    custom_image = torchvision.io.read_image(str(img_path)).type(torch.float32) /255
    
    print(custom_image.dtype) 
    print(custom_image.shape)

    # Führen Sie die Vorhersage mit dem Modell durch
    with torch.inference_mode():
        output_logits = predictor(custom_image.unsqueeze(0))
        print(output_logits)
        probabilities = torch.softmax(output_logits, dim=1)
        print(probabilities)
        pred_label = torch.argmax(probabilities, dim=1).item()
        print(pred_label)
        pred_class = class_names[pred_label]
        probability = probabilities.max()*100

    return pred_class, probability
