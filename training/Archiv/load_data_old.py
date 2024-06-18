
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

def load_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    #val_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader#, val_loader, len(train_dataset.classes)

data_dir = r"C:\Users\nme.AD\Desktop\Festo_Dataset\all_pictures"
data = load_data(data_dir)


print(data)