import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import random


input_dir = 'D:/Ronaldo/Nueva carpeta/archive/nuevo_train/train'
output_dir = 'D:/Ronaldo/Nueva carpeta/archive/nuevo_train/test'

Path(output_dir).mkdir(parents=True, exist_ok=True)
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

num_images_to_move = int(0.2 * len(image_files))
selected_images = random.sample(image_files, num_images_to_move)

for image_file in selected_images:
    input_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, image_file)

    shutil.move(input_path, output_path)

print("Proceso completado. Se movió el 20% de las imágenes al nuevo directorio.")
