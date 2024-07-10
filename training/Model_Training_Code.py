"""
Dieses Skript dient zur Vorbereitung von Bilddaten für das Training eines Klassifikationsmodells.
Es unterteilt die Bilder in Trainings-, Test- und Validierungssets, wendet Transformationen an
und lädt die Daten in PyTorch DataLoader zur weiteren Verarbeitung.
"""

import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch
import torchvision
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random


"""
Die folgenden Zeilen sind relevant, wenn das Modell auf Google Colab trainiert werden soll.
Sie stellen eine Verbindung zum Google Drive her, um auf die Daten zuzugreifen.
from google.colab import drive
drive.mount('/content/drive')
"""

# Funktion zum Zählen der Bilder und Ordner in einem Verzeichnis
def Bilder_im_Ordner(dir_path):
    for dirpath, dirnames, filename in os.walk(dir_path):
        print(f"Es gibt {len(dirnames)} Ordner und {len(filename)} Bilder in {dirpath}")

############################################ Aufteilung der Daten ###########################################################################
# Pfade für die Aufteilung der Daten
main_data_path = Path('/content/Gefiltert')
ziel_data_path = Path('/content/data_1')
train_folder = ziel_data_path / 'train'
test_folder = ziel_data_path / 'test'
#val_folder = ziel_data_path / 'val'

# Verhältnis der Aufteilung in Trainings- und Testdaten
train_ratio = 0.85
test_ratio = 0.15
#val_ratio = 0.15

# Erstellen der Zielordner
for folder in [train_folder, test_folder]:#, val_folder]:
    os.makedirs(folder, exist_ok=True)

# Funktion zum Leeren eines Ordners
def empty_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# Leeren der Zielordner
for folder in [train_folder, test_folder]:#, val_folder]:
    empty_folder(folder)

# Funktion zum Verschieben von Bildern in einen Zielordner
def move_images(images, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for img in images:
        shutil.copy(img, os.path.join(target_folder, img.name))

# Funktion zum Aufteilen der Bilder einer Klasse in Trainings- und Testdaten
def bilder_in_klassen_trennen(klasse_name, final_name):
    klasse_images = []
    for licht_folder in licht_folders:
        klasse_path = main_data_path / licht_folder / klasse_name
        if klasse_path.exists():
            klasse_images.extend([klasse_path / image for image in os.listdir(klasse_path)])

    train_images, test_images = train_test_split(klasse_images, test_size=(1-train_ratio), random_state=42)
    move_images(train_images, train_folder / final_name)
    move_images(test_images, test_folder / final_name)

# Ordner mit Bildern unter verschiedenen Beleuchtungsbedingungen
licht_folders = ['Licht_5_10', 'Licht_330_370', 'Licht_820_860']

# Dictionary zur Zuordnung der ursprünglichen Klassennamen zu den finalen Namen
klassen = {
    'Good_Parts': 'Good_Parts_all',
    'Filled_Parts': 'Filled_Parts_all',
    'Defective_Parts': 'Defective_Parts_all',
    'Defective_Filled_Parts': 'Defective_Parts_all'
}

# Aufteilung der Bilder in Trainings- und Testdaten für jede Klasse
for klasse_name, final_name in tqdm(klassen.items(), desc="Verarbeite Klassen"):
    bilder_in_klassen_trennen(klasse_name, final_name)

#######################################################################################################################

# Installation der Bibliotheken torchmetrics und mlxtend, falls nicht vorhanden
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher"
except:
    !pip install -q torchmetrics -U mlxtend
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")

# Import der ConfusionMatrix aus torchmetrics und plot_confusion_matrix aus mlxtend
# Diese werden später verwendet, um die Leistung des Modells zu evaluieren
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
#######################################################################################################################

# Pfade für Trainings-, Test- und Evaluierungsdaten (müssen noch definiert werden)
train_dir = ''
test_dir = ''
eval_dir = ''

#######################################################################################################################

# Definition der Transformationen für Trainings- und Testdaten
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.4),
    transforms.RandomRotation(degrees=(0, 360))
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

#######################################################################################################################

# Laden der Bilder aus den Ordnern als ImageFolder-Objekte
# Die Ordnernamen werden als Klassenlabels verwendet
train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
eval_data = datasets.ImageFolder(root=eval_dir, transform=test_transform)

# Anzeigen der Klassennamen
klassen_Namen = train_data.classes
print(klassen_Namen)

#######################################################################################################################

# Aufteilung der Bilder in Batches für das Training
BATCH_NUM = 16
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_NUM, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_NUM, shuffle=False)
eval_dataloader = DataLoader(dataset=eval_data, batch_size=BATCH_NUM, shuffle=False)

#######################################################################################################################

# Überprüfung, ob eine GPU verfügbar ist und Festlegen des Geräts für das Training
device = "cuda" if torch.cuda.is_available() else "cpu"

#######################################################################################################################
############################ Training und Evaluierungsfunktionen ######################################################
"""
Die folgenden Funktionen können verwendet werden, um ein Modell zu trainieren und zu evaluieren. 
Sie sind allgemein gehalten und können ohne Anpassungen für verschiedene Modelle und Datensätze eingesetzt werden. 
Kopieren Sie einfach den Code und rufen Sie die Funktionen mit den entsprechenden Parametern auf.

"""
def train_step(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str='cuda'
               ):
    """
    Führt einen Trainingsschritt für das gegebene Modell durch.
    
    Args:
        model: Das zu trainierende PyTorch-Modell.
        data_loader: Der DataLoader, der die Trainingsdaten bereitstellt.
        loss_fn: Die Verlustfunktion für das Training.
        optimizer: Der Optimierer für das Training.
        device: Das Gerät (CPU oder GPU), auf dem das Training durchgeführt wird.
        
    Returns:
        train_loss: Der durchschnittliche Trainingsverlust für diesen Schritt.
        train_acc: Die durchschnittliche Trainingsgenauigkeit für diesen Schritt.
        train_precision: Die durchschnittliche Präzision für diesen Schritt.
        train_recall: Die durchschnittliche Trefferquote (Recall) für diesen Schritt.
        train_f1: Der durchschnittliche F1-Score für diesen Schritt.
    """

  # Das Modell in den Trainingsmodus setzen
  model.train()
  # Train loss und Train accuracy einrichten
  train_loss, train_acc = 0, 0
  precision = torchmetrics.Precision(task="multiclass", num_classes=len(klassen_Namen), average='macro').to(device)
  recall = torchmetrics.Recall(task="multiclass",num_classes=len(klassen_Namen), average='macro').to(device)
  f1 = torchmetrics.F1Score(task="multiclass",num_classes=len(klassen_Namen), average='macro').to(device)


  # Eine Schleife hinzufügen, um durch die Trainingsbatches zu iterieren
  for batch, (IMAGE_BATCH,LABEL_BATCH) in enumerate(data_loader):
    IMAGE_BATCH, LABEL_BATCH = IMAGE_BATCH.to(device), LABEL_BATCH.to(device)

    # 1. Forward pass
    y_pred=model(IMAGE_BATCH)

    # 2. loss berechnen
    loss = loss_fn(y_pred,LABEL_BATCH)
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Acc berechnen
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1),dim=1)
    train_acc += (y_pred_class==LABEL_BATCH).sum().item()/len(y_pred)
    precision(y_pred_class, LABEL_BATCH)
    recall(y_pred_class, LABEL_BATCH)
    f1(y_pred_class, LABEL_BATCH)

  # Den gesamten Loss und die Acc durch die Länge des Dataloader teilen
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  train_precision = precision.compute()
  train_recall = recall.compute()
  train_f1 = f1.compute()

  precision.reset()
  recall.reset()
  f1.reset()


  #print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}%")
  return train_loss, train_acc, train_precision, train_recall, train_f1

def test_step(model:torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str = 'cuda'
              ):
    """
    Führt einen Testschritt für das gegebene Modell durch.
    
    Args:
        model: Das zu testende PyTorch-Modell.
        data_loader: Der DataLoader, der die Testdaten bereitstellt.
        loss_fn: Die Verlustfunktion für den Test.
        device: Das Gerät (CPU oder GPU), auf dem der Test durchgeführt wird.
        
    Returns:
        test_loss: Der durchschnittliche Testverlust für diesen Schritt.
        test_acc: Die durchschnittliche Testgenauigkeit für diesen Schritt.
        test_precision: Die durchschnittliche Präzision für diesen Schritt.
        test_recall: Die durchschnittliche Trefferquote (Recall) für diesen Schritt.
        test_f1: Der durchschnittliche F1-Score für diesen Schritt.
    """

  # Das Modell in den Evaluierungsmodus setzen
  model.eval()

  precision = torchmetrics.Precision(task="multiclass", num_classes=len(klassen_Namen), average='macro').to(device)
  recall = torchmetrics.Recall(task="multiclass", num_classes=len(klassen_Namen), average='macro').to(device)
  f1 = torchmetrics.F1Score(task="multiclass", num_classes=len(klassen_Namen), average='macro').to(device)


  # Loss und Acc einrichten
  test_loss, test_acc = 0, 0

  # Inferenzmodus aktivieren
  with torch.inference_mode():
    for X_test, y_test in data_loader:
      X_test, y_test = X_test.to(device), y_test.to(device)
      # Forward pass
      test_pred_logits = model(X_test)

      # Loss berechnen
      loss = loss_fn(test_pred_logits,y_test)
      test_loss += loss.item()
      # Acc berechnen
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels==y_test).sum().item()/len(test_pred_labels))

      precision(test_pred_labels, y_test)
      recall(test_pred_labels, y_test)
      f1(test_pred_labels, y_test)


  # Den gesamten Loss und die Acc durch die Länge des Dataloader teilen
  test_loss /= len(data_loader)
  test_acc /= len(data_loader)
  test_precision = precision.compute()
  test_recall = recall.compute()
  test_f1 = f1.compute()

  #print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")
  return test_loss, test_acc, test_precision, test_recall, test_f1

def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn:torch.nn.Module=nn.CrossEntropyLoss(),
          epochs:int=5,
          device:str='cuda'
          ):
    """
    Trainiert das gegebene Modell für eine bestimmte Anzahl von Epochen.
    
    Args:
        model: Das zu trainierende PyTorch-Modell.
        train_dataloader: Der DataLoader für die Trainingsdaten.
        test_dataloader: Der DataLoader für die Testdaten.
        optimizer: Der Optimierer für das Training.
        loss_fn: Die Verlustfunktion für das Training (Standard: CrossEntropyLoss).
        epochs: Die Anzahl der Trainingsepochen (Standard: 5).
        device: Das Gerät (CPU oder GPU), auf dem das Training durchgeführt wird.
        
    Returns:
        results: Ein Dictionary mit den Trainings- und Testergebnissen für jede Epoche.
    """


  # 2. Leeres Dict erstellen
  results = {"train_loss": [],
             "train_acc":[],
             "train_precision": [],
             "train_recall": [],
             "train_f1": [],
             "test_loss": [],
             "test_acc": [],
             "test_precision": [],
             "test_recall": [],
             "test_f1": []
             }

  # 3. Durch Training und Test für eine Anzahl von Epochen iterieren
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(model=model,
                                                                                data_loader=train_dataloader,
                                                                                loss_fn=loss_fn,
                                                                                optimizer=optimizer)

    test_loss, test_acc,test_precision, test_recall, test_f1  = test_step(model=model,
                                                                          data_loader=test_dataloader,
                                                                          loss_fn=loss_fn)
    # 4. Ausgeben, was gerade passiert
    print(f"Epoch: {epoch} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f} | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
          f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}")

    # 5. Die Ergebnisse aktualisieren
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["train_precision"].append(train_precision)
    results["train_recall"].append(train_recall)
    results["train_f1"].append(train_f1)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    results["test_precision"].append(test_precision)
    results["test_recall"].append(test_recall)
    results["test_f1"].append(test_f1)


  return results

def eval_model(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module):
    """
    Evaluiert das gegebene Modell auf den bereitgestellten Daten.
    
    Args:
        model: Das zu evaluierende PyTorch-Modell.
        data_loader: Der DataLoader, der die Evaluierungsdaten bereitstellt.
        loss_fn: Die Verlustfunktion für die Evaluierung.
        
    Returns:
        Ein Dictionary mit dem Modellnamen, dem durchschnittlichen Verlust und der durchschnittlichen Genauigkeit.
    """
    
  # Das Modell in den Evaluierungsmodus setzen
  model.eval()

  eval_loss, eval_acc = 0, 0
  with torch.inference_mode():
    for X,y in data_loader:
      X, y = X.to(device), y.to(device)

      # Forward pass
      eval_pred_logit = model(X)

      # loss and acc berechnen
      loss = loss_fn(eval_pred_logit,y)
      eval_loss += loss.item()

      eval_pred_class = torch.argmax(torch.softmax(eval_pred_logit,dim=1), dim=1)
      eval_acc += (eval_pred_class==y).sum().item() / len(eval_pred_logit)

    eval_loss /= len(data_loader)
    eval_acc /= len(data_loader)

    return {"model_name:": model.__class__.__name__ ,
            "model_loss:":eval_loss,
            "model_acc:": eval_acc}

########################################## Model Herunterladen ####################################
# Laden des vortrainierten ResNet101-Modells
model_ResNet101 = models.resnet101(pretrained=True)


###################################### Summary Library importieren ###############################
# Installation und Import der torchinfo-Bibliothek für Modellinformationen
try:
    import torchinfo
except:
    !pip install torchinfo
    from torchinfo import summary

# Zusammenfassung des Modells anzeigen
summary(model=model_test, input_size=[16, 3, 224, 224])


########################### Anpassung der letzten Schicht #################
# Anpassung der letzten fully connected Schicht an die Anzahl der Klassen
num_ftrs = model_ResNet101.fc.in_features
model_ResNet101.fc = nn.Linear(num_ftrs, len(klassen_Namen))
model_test = model_ResNet101.to(device)

# Definition der Verlustfunktion und des Optimierers
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ResNet101.parameters(),
                             lr=0.000001)


############################## Model trainieren ##################################
# Setzen eines Seeds für Reproduzierbarkeit
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Definition der Trainingsparameter und Durchführung des Trainings
EPOCHS_NUM = 150
model_test_result = train(model=model_test, 
                          train_dataloader=train_dataloader, 
                          test_dataloader=test_dataloader, 
                          loss_fn=loss_fn, 
                          optimizer=optimizer, 
                          epochs=EPOCHS_NUM, 
                          device='cuda')


####################### Ploten loss und acc Kurven ########################

from typing import Dict, List
import pickle
def plot_loss_and_acc_curve(results: Dict[str, List[float]]):
    epochs = range(1, len(results["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(15, 7))


    ax1.plot(epochs, results["train_loss"], label="Train Loss", color="orange")
    ax1.plot(epochs, results["test_loss"], label="Test Loss", color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, results["train_acc"], label="Train Accuracy", color="orange")
    ax2.plot(epochs, results["test_acc"], label="Test Accuracy", color="blue")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")

    plt.title("Training Loss and Accuracy")
    plt.show()


############################ Ploten von Confisuen Matrix ###########################

# Plott confusion matrix
from typing import Dict, List

def plot_model_confusion_Matrix(model:torch.nn.Module,
                                data_loader:torch.utils.data.DataLoader,
                                data,
                                class_names:List
                                ):
  # 1. Vorhersagen mit dem trainierten Modell machen
  y_preds = []
  model.eval()
  with torch.inference_mode():
   for X, y in tqdm(data_loader, desc="Vorhersage machen"):
      X, y = X.to(device), y.to(device)

      y_logit = model(X)

      y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

      y_preds.append(y_pred.cpu())

  y_pred_tensor = torch.cat(y_preds)
  # 2. Instanz der Confusion Matrix einrichten und Vorhersagen mit den Zielen vergleichen
  confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')

  if not isinstance(data.targets, torch.Tensor):
      data_tensor = torch.tensor(data.targets)

  confmat_tensor = confmat(preds=y_pred_tensor,
                         target=data_tensor
                         )


  fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(5, 6)
    );

plot_model_confusion_Matrix(model=model_test,
                            data_loader=test_dataloader,
                            data=test_data,
                            class_names=klassen_Namen)

################################ Datenvertreilung ploten #######################

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


custom_feature_image_dataset = CustomImageFolder('/content/Gefiltert', transform=transform)

custom_feature_dataloader = DataLoader(custom_feature_image_dataset, batch_size=8, shuffle=False)

features, labels, image_paths = [], [], []
for inputs, classes, paths in custom_feature_dataloader:
    outputs = model_test(inputs.to(device))
    features.extend(outputs.detach().cpu().numpy())
    labels.extend(classes.numpy())
    image_paths.extend(paths)

features = np.array(features)
labels = np.array(labels)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.figure(figsize=(10, 8))
for class_index in np.unique(labels):
    plt.scatter(reduced_features[labels == class_index, 0], reduced_features[labels == class_index, 1], label=f'{klassen_Namen[class_index]}', alpha=0.5)
plt.legend()
plt.title('Datenpunkte Verteilung')
plt.axis(False)
plt.show()



####################### Plotten von Verlust- und Genauigkeitskurven ########################
from typing import Dict, List
import pickle

def plot_loss_and_acc_curve(results: Dict[str, List[float]]):
    """
    Plottet die Verlust- und Genauigkeitskurven für das Training und den Test.
    
    Args:
        results: Ein Dictionary mit den Trainings- und Testergebnissen.
    """
    epochs = range(1, len(results["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Verlustskurven plotten
    ax1.plot(epochs, results["train_loss"], label="Train Loss", color="orange")
    ax1.plot(epochs, results["test_loss"], label="Test Loss", color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    # Genauigkeitskurven plotten
    ax2 = ax1.twinx()
    ax2.plot(epochs, results["train_acc"], label="Train Accuracy", color="orange", linestyle="--")
    ax2.plot(epochs, results["test_acc"], label="Test Accuracy", color="blue", linestyle="--")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")

    plt.title("Training Loss and Accuracy")
    plt.show()

############################ Plotten der Konfusionsmatrix ###########################
from typing import Dict, List

def plot_model_confusion_Matrix(model: torch.nn.Module,
                                data_loader: torch.utils.data.DataLoader,
                                data,
                                class_names: List):
    """
    Erstellt und plottet die Konfusionsmatrix für das gegebene Modell und die Testdaten.
    
    Args:
        model: Das trainierte PyTorch-Modell.
        data_loader: Der DataLoader mit den Testdaten.
        data: Das Dataset-Objekt.
        class_names: Eine Liste mit den Namen der Klassen.
    """
    # Vorhersagen mit dem trainierten Modell machen
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Vorhersage machen"):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)

    # Konfusionsmatrix erstellen
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    if not isinstance(data.targets, torch.Tensor):
        data_tensor = torch.tensor(data.targets)
    confmat_tensor = confmat(preds=y_pred_tensor, target=data_tensor)

    # Konfusionsmatrix plotten
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(5, 6)
    )

# Konfusionsmatrix für das Testset plotten
plot_model_confusion_Matrix(model=model_test,
                            data_loader=test_dataloader,
                            data=test_data,
                            class_names=klassen_Namen)

################################ Datenverteilung plotten #######################
class CustomImageFolder(datasets.ImageFolder):
    """
    Erweiterte ImageFolder-Klasse, die auch den Dateipfad zurückgibt.
    """
    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# Erstellen des angepassten Datasets und DataLoaders
custom_feature_image_dataset = CustomImageFolder('/content/Gefiltert', transform=transform)
custom_feature_dataloader = DataLoader(custom_feature_image_dataset, batch_size=8, shuffle=False)

# Extrahieren der Features und Labels
features, labels, image_paths = [], [], []
for inputs, classes, paths in custom_feature_dataloader:
    outputs = model_test(inputs.to(device))
    features.extend(outputs.detach().cpu().numpy())
    labels.extend(classes.numpy())
    image_paths.extend(paths)

features = np.array(features)
labels = np.array(labels)

# PCA zur Dimensionsreduktion
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Plotten der reduzierten Features
plt.figure(figsize=(10, 8))
for class_index in np.unique(labels):
    plt.scatter(reduced_features[labels == class_index, 0], 
                reduced_features[labels == class_index, 1], 
                label=f'{klassen_Namen[class_index]}', 
                alpha=0.5)
plt.legend()
plt.title('Datenpunkte Verteilung')
plt.axis('off')
plt.show()

######################################### Daten Visualisieren #############################

# Seed für Reproduzierbarkeit setzen
random.seed(30)

# Liste aller Bildpfade erstellen
image_path_list = list(image_path.glob('*/*/*.jpg'))

# Zufälliges Bild auswählen
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem

# Bild öffnen und Informationen ausgeben
img = Image.open(random_image_path)
print(f"Zufälliger Bildpfad: {random_image_path}")
print(f"Bildklasse: {image_class}")
print(f"Bildhöhe: {img.height}")
print(f"Bildbreite: {img.width}")

# Bild als Numpy-Array anzeigen
img_as_array = np.asarray(img)
plt.figure(figsize=(5,5))
plt.imshow(img_as_array)
plt.title(image_class)
plt.axis(False)

def plot_transformed_images(image_paths: list, transform, n=4, seed=None):
    """
    Plottet originale und transformierte Versionen von zufällig ausgewählten Bildern.
    
    Args:
        image_paths: Liste der Bildpfade
        transform: Anzuwendende Transformation
        n: Anzahl der zu plottenden Bilder
        seed: Seed für die Zufallsauswahl
    """
    if seed:
        random.seed(seed)
    else:
        random.seed(42)
    
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            fig.suptitle(f"Klasse: {image_path.parent.stem}")
            
            # Originalbild plotten
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nGröße: {f.size}")
            ax[0].axis(False)
            
            # Transformiertes Bild plotten
            transformed_image = transform(f)
            ax[1].imshow(transformed_image.permute(1, 2, 0))
            ax[1].set_title(f"Transformierte Größe\n{transformed_image.shape}")
            ax[1].axis(False)

