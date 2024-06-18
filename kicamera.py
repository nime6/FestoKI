#!/usr/bin/env python3
# coding=utf8
from settings import img_dir
from picamera2 import Picamera2
from PIL import Image
import numpy as np
import os
import time

from datetime import datetime




import io

data = io.BytesIO()
picam = Picamera2()

def setup_camera():
    #Setup PICAMERA
    picam.options["quality"] = 100
    config = picam.create_still_configuration()
    picam.configure(config)
    picam.start()
    #time.sleep(2)
    return picam

def take_picture():
    """#Take picture
    picam.capture_file(data, format='jpeg')
    data.seek(0)
    img = Image.open(data)
    #Crop picture
    BOX = [428,160,1196,928]
    img = img.crop(BOX)
    """
    
     #Take picture

    img_array = picam.capture_array("main")
    img = Image.fromarray(img_array)

    #picam.capture_file(data, format='jpeg')
    #data.seek(0)
    #img = Image.open(data)
    
    #Crop picture
    box = [428,210,1196,988]
    #box = [428,160,1196,928]
    img = img.crop(box)


    return img

def save_img(in_img, img_dir=img_dir):
    current_datetime = str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    file_prefix = f"{current_datetime}-res"
    file_path = os.path.join(img_dir, file_prefix + ".jpg")
    in_img.save(file_path)
    return file_path

def resize_img(in_img, size):
    out_img = in_img.resize(size)
    return out_img

def get_img():
    start = time.time()

    #setup_camera() #moved to main

    img = take_picture()
    img224 = resize_img(img, [224,224])
    img_path = save_img(img224)
    
    #img299 = resize_img(img, [299,299])
    #save_img(img299)
    #img513 = resize_img(img, [513,513])
    #save_img(img513)


    end = time.time()
    #print("img timer:")
    #print(end - start)
    
    np_img = np.array(img224)
    return np_img, img_path

def close_cam():
    picam.close()
    print("Picam closed")
