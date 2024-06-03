import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request
from model import Predictor



class_names = ['Defect', 'Filled','Good']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image():
    pass
    #if path: load img from path
    #if array: load img from array
    #...
    #return image as tensor
    
    
    
# Function to perform inference on an image
def predict_image(img, model_path):
    model = Predictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    #image = Image.open(image_path).convert("RGB")
    image_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    print(image_tensor.shape)
    #with torch.no_grad():
    output_logits = model(image_tensor)
    probability = torch.softmax(output_logits,  dim=1)
    pred_label = torch.argmax(probability, dim = 1)
    pred_class = class_names[pred_label]
    
    
    return pred_class, probability

# Function to display image, true label, predicted label, and probability
def display_results(image_path, true_label, predicted_label, probability):
    image = Image.open(image_path)
    print("True Label:", true_label)
    print("Predicted Label:", class_names[predicted_label.item()])
    print("Probability:", probability.item())
    #image.show()

def display_multiclass_results():
    pass
    

        
if __name__ == "__main__":
    # Input image path
    image_path = r"/home/pi/code/G_Dataset/Licht_820_840/Filld_Parts/GS-Camera-2024-01-23--10-48-18-res513.jpg"  # Replace with the path to your input image

    # True label (optional, provide if known)
    true_label = 1  # Replace with the true label of the input image
    # Perform inference
    predicted_label, probability = predict_image(img, model_path)
    # Display results
    display_results(image_path, true_label, predicted_label, probability)

 
    
