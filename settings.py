from datetime import datetime

#Config
current_datetime = str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

img_dir = r"/home/pi/pictures/GS-Camera-"
file_prefix = f"{current_datetime}-res"