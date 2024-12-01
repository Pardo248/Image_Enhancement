import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Función para aumentar la resolución de una imagen
def upscale_image(model, lr_image):
    lr_image = np.expand_dims(lr_image, axis=0)  # Añadir dimensión para batch
    sr_image = model.predict(lr_image)[0]  # Quitar batch
    sr_image = np.clip(sr_image, 0, 1)  # Asegurar valores entre 0 y 1
    return sr_image

if __name__ == "__main__":
    # Cargar el modelo previamente entrenado
    model = tf.keras.models.load_model('super_resolution_model.h5')

    # Cargar una imagen de baja resolución para probar
    lr_image_path = 'ruta/a/imagen_de_prueba.jpg'
    lr_image = load_img(lr_image_path)
    lr_image = img_to_array(lr_image) / 255.0

    # Realizar la predicción
    sr_image = upscale_image(model, lr_image)

    # Mostrar los resultados
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Low Resolution")
    plt.imshow(lr_image)

    plt.subplot(1, 3, 2)
    plt.title("Super Resolution")
    plt.imshow(sr_image)

    # Para la imagen de alta resolución (HR), puedes cargarla y mostrarla si la tienes
    hr_image_path = 'ruta/a/imagen_de_alta_resolucion.jpg'
    hr_image = load_img(hr_image_path)
    hr_image = img_to_array(hr_image) / 255.0
    plt.subplot(1, 3, 3)
    plt.title("High Resolution")
    plt.imshow(hr_image)

    plt.show()
