import sys
import time
import RPi.GPIO as GPIO
import smbus
import torch
import torch.nn as nn
from torchvision import transforms
import kicamera
from inference import predict_image_multiclass, Predictor
from settings import model_path
import threading
import gradio as gr
import base64
import os
import matplotlib.pyplot as plt
from PIL import Image

predictor = torch.jit.load(model_path)
predictor.eval()

# Globale Variablen
PHMAKE = 0
stepstomove = 270000*16
stepstomove_remaining = stepstomove
latest_prediction = {"class": None, "probability": None, "image_path": None}
stop_flag = threading.Event()
update_gui = threading.Event()

# GPIO Setup
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

LIBA = 11
STENA = 31
STPUL = 33
STDIR = 35

STSTEP_ANGLE = 1.8 / 16
STRAMP_LENGTH = 300*16
STMIN_RPM = 60
STMAX_RPM = 300

STstepsPerRevolution = 1440 / STSTEP_ANGLE
STmintime4step = 1 / (STMAX_RPM / 60 * STstepsPerRevolution) / 2
STmaxtime4step = 1 / (STMIN_RPM / 60 * STstepsPerRevolution) / 2
STrampSlope = (STmaxtime4step - STmintime4step) / STRAMP_LENGTH
STtime4step = STmaxtime4step

GPIO.setup(LIBA, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
GPIO.setup(STDIR, GPIO.OUT)
GPIO.setup(STPUL, GPIO.OUT)
GPIO.setup(STENA, GPIO.OUT)

STDIR_Left = GPIO.HIGH
STDIR_Right = GPIO.LOW
STEnable = GPIO.LOW
STDisable = GPIO.HIGH

def doIfFalling(channel):
    global PHMAKE, stepstomove, stepstomove_remaining
    PHMAKE = 1   
    stepstomove = (970*16) + 180
    stepstomove_remaining = stepstomove
    print("GPIO Falling to LOW - ", str(stepstomove), "Steps remaining")

GPIO.add_event_detect(LIBA, GPIO.FALLING, callback=doIfFalling, bouncetime=100)

def quality_assurance():
    global PHMAKE, stepstomove, stepstomove_remaining, latest_prediction
    img, img_path = kicamera.get_img()
    print(f"Image captured: {img_path}")
    pred_class, probability = predict_image_multiclass(img_path, predictor)
    print(f"Prediction: {pred_class}, Probability: {probability}")
    latest_prediction = {
        "class": pred_class,
        "probability": probability,
        "image_path": img_path
    }
    update_gui.set()  # Signal GUI update
    return pred_class, probability

def tuersteher():
    DEVICE_BUS = 1
    DEVICE_ADDR = 0x10
    bus = smbus.SMBus(DEVICE_BUS)
    bus.write_byte_data(DEVICE_ADDR, 1, 0xFF)
    time.sleep(3)  
    bus.write_byte_data(DEVICE_ADDR, 1, 0x00)

def motor_control():
    global stepstomove_remaining, STtime4step, STPUL, STRAMP_LENGTH, stepstomove, STmaxtime4step, STmintime4step, STrampSlope
    while not stop_flag.is_set():
        if stepstomove_remaining > 0:
            if (stepstomove > 2 * STRAMP_LENGTH):
                if (stepstomove_remaining < STRAMP_LENGTH):
                    STtime4step = STmaxtime4step - (STrampSlope*stepstomove_remaining)
                elif(stepstomove_remaining > stepstomove - STRAMP_LENGTH):
                    STtime4step = (5 * STtime4step + STmaxtime4step - (STrampSlope*(stepstomove - stepstomove_remaining))) / 6
                else:
                    STtime4step = STmintime4step        
            GPIO.output(STPUL, GPIO.LOW)
            time.sleep(STtime4step / 2)
            GPIO.output(STPUL, GPIO.HIGH)
            time.sleep(STtime4step / 2)
            stepstomove_remaining -= 1

def main_loop():
    global PHMAKE, stepstomove, stepstomove_remaining
    while not stop_flag.is_set():
        GPIO.output(STENA, STEnable)
        if stepstomove_remaining <= 0:
            stepstomove_remaining = 0
        if stepstomove_remaining == 0 and PHMAKE == 1:
            time.sleep(0.5)
            PHMAKE = 0    
            pred_class, probability = quality_assurance()
            if pred_class == "Defect":
                threading.Thread(target=tuersteher).start()
            time.sleep(1)
            stepstomove = 270000*16
            stepstomove_remaining = stepstomove
        time.sleep(0.01)

def start_system():
    print("Starting system...")
    stop_flag.clear()
    threading.Thread(target=motor_control, daemon=True).start()
    threading.Thread(target=main_loop, daemon=True).start()
    return "System gestartet"

def stop_system():
    print("Stopping system...")
    stop_flag.set()
    GPIO.output(STENA, STDisable)
    return "System gestoppt"

def update_prediction():
    update_gui.wait()
    update_gui.clear()
    if latest_prediction["image_path"]:
        img = Image.open(latest_prediction["image_path"])
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Vorhersage: {latest_prediction['class']}\nWahrscheinlichkeit: {latest_prediction['probability']:.2f}%")
        return plt, f"Letztes Bild: {latest_prediction['image_path']}"
    return None, "Kein Bild verfügbar"

def embed_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            file_cont = image_file.read()
        image_type = os.path.splitext(image_path)[1].lower()
        if image_type in [".jpg", ".jpeg"]:
            img_type = "image/jpeg"
        elif image_type == ".png":
            img_type = "image/png"
        elif image_type == ".gif":
            img_type = "image/gif"
        else:
            raise ValueError(f"Nicht unterstütztes Bildformat")
        encoded_string = base64.b64encode(file_cont).decode()
        return f"data:{img_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error in embed_image: {e}")
        return ""

def main():
    kicamera.setup_camera()

    logos_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: white; flex-wrap: wrap;">
        <img src="{embed_image('/home/pi/Desktop/FestoKI/GUI/DMB_LOGO.png')}" alt="Left Logo" style="height: 60px; margin: 5px;">
        <h1 style="font-size: 3em; text-align: center; flex-grow: 1; width: 100%; order: 1;">Automatisches Bildvorhersage-System</h1>
        <img src="{embed_image('/home/pi/Desktop/FestoKI/GUI/uni_pd.jpg')}" alt="Right Logo" style="height: 60px; margin: 5px;">
    </div>
    """

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(logos_html)
        
        gr.Markdown("""
        ## Anleitung
        1. Klicken Sie auf 'System starten', um die Anlage zu aktivieren.
        3. Klicken Sie auf 'System stoppen', um die Anlage zu deaktivieren.
        """)
        
        with gr.Row():
            start_btn = gr.Button("System starten")
            stop_btn = gr.Button("System stoppen")
        
        status_output = gr.Textbox(label="Status", value="System gestoppt")
        
        with gr.Row():
            image_output = gr.Plot(label="Aktuelles Bild mit Vorhersage")
        with gr.Row():
            prediction_output = gr.Textbox(label="Letzte Vorhersage")
        
        start_btn.click(start_system, outputs=[status_output])
        stop_btn.click(stop_system, outputs=[status_output])
        
        demo.load(update_prediction, outputs=[image_output, prediction_output], every=1)

    demo.launch()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Programm wurde durch Benutzer beendet.")
    finally:
        stop_flag.set()
        GPIO.cleanup()
        kicamera.close_cam()
