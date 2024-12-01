import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Función para cargar imágenes desde las carpetas
def load_images(lr_folder):
    lr_images = []

    # Cargar imágenes de baja calidad (LR)
    for filename in os.listdir(lr_folder):
        file_path = os.path.join(lr_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_lr = load_img(file_path)
            img_lr = img_to_array(img_lr) / 255.0
            lr_images.append(img_lr)

    return np.array(lr_images)

# Función para aumentar la resolución de una imagen
def upscale_image(model, lr_image):
    lr_image = np.expand_dims(lr_image, axis=0)  # Añadir dimensión para batch
    sr_image = model.predict(lr_image)[0]  # Quitar batch
    sr_image = np.clip(sr_image, 0, 1)  # Asegurar valores entre 0 y 1
    return sr_image

if __name__ == "__main__":
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model("super_resolution_model.h5")

    # Directorio de imágenes de baja resolución (LR)
    lr_folder = r"D:\machin_proyecto\mala_calidad"  # Cambia esta ruta si es necesario

    # Cargar las imágenes de baja resolución
    lr_images = load_images(lr_folder)

    # Probar con una imagen de baja resolución
    test_lr_image = lr_images[0]
    sr_image = upscale_image(model, test_lr_image)

    # Mostrar resultados
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Low Resolution")
    plt.imshow(test_lr_image)

    plt.subplot(1, 3, 2)
    plt.title("Super Resolution")
    plt.imshow(sr_image)

    # Cargar una imagen de alta resolución para la comparación (opcional)
    hr_image = load_img(r"D:\machin_proyecto\buena_calidad\example.jpg")  # Cambia esta ruta si es necesario
    hr_image = img_to_array(hr_image) / 255.0
    plt.subplot(1, 3, 3)
    plt.title("High Resolution")
    plt.imshow(hr_image)

    plt.show()
