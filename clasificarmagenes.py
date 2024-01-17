import os
import cv2
import numpy as np
from pathlib import Path
import shutil

input_dir = 'D:/Ronaldo/Nueva carpeta/archive/cars_train/cars_train'
output_dir = 'D:/Ronaldo/Nueva carpeta/archive/nuevo_train'

Path(output_dir).mkdir(parents=True, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

selected_images = image_files[:500]
desired_size = (255, 255)

for image_file in selected_images:
    image_path = os.path.join(input_dir, image_file)
    img = cv2.imread(image_path)

    img_resized = cv2.resize(img, desired_size)
    img_normalized = cv2.normalize(img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, img_normalized)

print("Proceso completado. Imágenes procesadas y guardadas en el nuevo directorio.")
