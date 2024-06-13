import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import Predictor


class_names = ['Defect', 'Filled','Good']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image_as_tensor(image_input):
    """
    Load an image from a NumPy array, PIL image, or file path and return as a PyTorch tensor.

    Parameters:
    - image_input: NumPy array, PIL Image, or file path (str).

    Returns:
    - image_tensor: PyTorch tensor.
    """
    # Define a transform to convert the image to a PyTorch tensor
    transform = transforms.ToTensor()

    if isinstance(image_input, np.ndarray):
        # If the input is a NumPy array, convert it to a PIL image first
        image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        # If the input is already a PIL image, use it directly
        image = image_input
    elif isinstance(image_input, str):
        # If the input is a file path, load the image from the file
        image = Image.open(image_input)
    else:
        raise TypeError("Unsupported image input type. Must be a NumPy array, PIL image, or file path.")

    # Apply the transform to convert the PIL image to a PyTorch tensor
    image_tensor = transform(image)

    return image_tensor
 
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
    true_label = 1  # Replace with the true label of the input image

    # Inference
    predicted_label, probability = predict_image(img, model_path)

    # Display results
    display_results(image_path, true_label, predicted_label, probability)

 
    
