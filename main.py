#!/usr/bin/env python
#coding: utf8 
import time
import RPi.GPIO as GPIO
import smbus
import kicamera
from inference import predict_image
from settings import model_path

def doIfFalling(channel):
    global stepstomove
    global stepstomove_remaining
    global PHMAKE
    PHMAKE = 1   
    stepstomove = 970*16
    stepstomove_remaining = stepstomove
    print("GPIO Falling to LOW - " , str(stepstomove), "Steps remaining")
            
def quality_assurance():
    global PHMAKE
    global stepstomove
    global stepstomove_remaining


    img = kicamera.get_img()

    pred_class, probability = predict_image(img, model_path)
    print( pred_class, probability )
    #print (f"taking photo, img size: {img.size}")

    return pred_class, probability

    
def tuersteher():
    DEVICE_BUS = 1
    DEVICE_ADDR = 0x10
    bus = smbus.SMBus(DEVICE_BUS)

    bus.write_byte_data(DEVICE_ADDR, 1, 0xFF) # Activate ejector and LED
    time.sleep(1)
    bus.write_byte_data(DEVICE_ADDR, 1, 0x00) # Deactivate ejector and LED
    
    """Bus Adresses
            bus.write_byte_data(DEVICE_ADDR, 1, 0x00)
            bus.write_byte_data(DEVICE_ADDR, 2, 0x00)
            bus.write_byte_data(DEVICE_ADDR, 3, 0x00)
            bus.write_byte_data(DEVICE_ADDR, 4, 0x00)
            sys.exit()
    """     

if __name__ == "__main__":
    kicamera.setup_camera()


    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    PHMAKE = 0
    # Light Barrier GPIO-Setup
    LIBA = 11
    # StepperDriver DM542 GPIO-Setup
    STENA = 31
    STPUL = 33
    STDIR = 35
    # Motor Setup
    STSTEP_ANGLE = 1.8 / 16#1.8 # degree
    STRAMP_LENGTH = 300*16 # steps
    STMIN_RPM = 60
    STMAX_RPM = 300
    # Step Frequency calculation
    STstepsPerRevolution = 1440 / STSTEP_ANGLE
    STmintime4step = 1 / (STMAX_RPM / 60 * STstepsPerRevolution) / 2
    STmaxtime4step = 1 / (STMIN_RPM / 60 * STstepsPerRevolution) / 2
    STrampSlope = (STmaxtime4step - STmintime4step) / STRAMP_LENGTH
    STtime4step = STmaxtime4step

    GPIO.setup(LIBA, GPIO.IN, pull_up_down = GPIO.PUD_OFF)
    GPIO.setup(STDIR, GPIO.OUT)
    GPIO.setup(STPUL, GPIO.OUT)
    GPIO.setup(STENA, GPIO.OUT)

    STDIR_Left = GPIO.HIGH
    STDIR_Right = GPIO.LOW
    STEnable = GPIO.LOW
    STDisable = GPIO.HIGH

    stepstomove = 270000*16
    stepstomove_remaining = stepstomove

    GPIO.add_event_detect(LIBA, GPIO.FALLING, callback = doIfFalling, bouncetime = 100)

    try:
        while 1:
            GPIO.output(STENA, STEnable)
    #        time.sleep(0.5)
            if stepstomove_remaining <= 0:
                stepstomove_remaining=0
            if stepstomove_remaining == 0 and PHMAKE == 1:
                PHMAKE = 0    

                #Start QA functionality_______________________________________________________________________________________________________
                pred_class, probability = quality_assurance()
                
                if pred_class == "Defect":
                    tuersteher()
        
                #End QA functionality_______________________________________________________________________________________________________
                time.sleep(1)
                stepstomove = 270000
                stepstomove_remaining = stepstomove
                
            if (stepstomove_remaining > 0):
                if (stepstomove > 2 * STRAMP_LENGTH):
                    if (stepstomove_remaining < STRAMP_LENGTH):
    #                    print ("-1-")
                       STtime4step = STmaxtime4step - (STrampSlope*stepstomove_remaining)
                    elif(stepstomove_remaining > stepstomove - STRAMP_LENGTH):
    #                    print ("-2-")
                       STtime4step = (5 * STtime4step + STmaxtime4step - (STrampSlope*(stepstomove - stepstomove_remaining))) / 6
                    else:
    #                   print ("-3-")
                       STtime4step = STmintime4step        
            # Schritt ausführen
                GPIO.output(STPUL, GPIO.LOW)
                time.sleep(STtime4step / 2)
                GPIO.output(STPUL, GPIO.HIGH)
                time.sleep(STtime4step / 2)
                stepstomove_remaining-=1
    #            print("stepstomove = ", str(stepstomove),"stepstomove_remaining = ", str(stepstomove_remaining), ", STtime4step = ", str(STtime4step))
                
                
    except KeyboardInterrupt:
    #    print ("STmaxtime4step = ",str(STmaxtime4step))
    #    print ("STmintime4step = ",str(STmintime4step))
        print ("Quit")
        GPIO.output(STENA, STDisable)
    #    GPIO.cleanup()
    
        kicamera.close_cam()


