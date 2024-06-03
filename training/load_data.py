#%%
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import math
#%%
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for label, folder in enumerate(self.classes):
            folder_path = os.path.join(self.root_dir, folder)
            for image_file in (os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_file)
                data.append((image_path, torch.tensor(label, dtype=float)))
                
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        #image = torch.tensor(np.array(image))
        if self.transform:
            image = self.transform(image)
        return image, label


# Example usage:
root_dir1 = r"C:\Users\nme.AD\Desktop\Festo_Dataset\all_pictures"

transform =     transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # You can add transforms here if needed

dataset = CustomDataset(root_dir1, transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, lengths=[math.ceil(0.7*len(dataset)), math.floor(0.3*len(dataset))])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
print(len(dataset))
# Now you can iterate over dataloader
for images, labels in train_loader:
    #print(images.shape)
    #print(labels)
    break
    # Your training/validation loop here
    pass

#%%