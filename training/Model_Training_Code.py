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


