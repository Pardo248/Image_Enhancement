import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Función para cargar imágenes desde las carpetas
def load_images(lr_folder):
    lr_images = []
    filenames = []

    # Cargar imágenes de baja calidad (LR)
    for filename in os.listdir(lr_folder):
        file_path = os.path.join(lr_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_lr = load_img(file_path)
            img_lr = img_to_array(img_lr) / 255.0  # Normalizar
            lr_images.append(img_lr)
            filenames.append(filename)

    return np.array(lr_images), filenames

# Función para aumentar la resolución de una imagen
def upscale_image(model, lr_image):
    lr_image = np.expand_dims(lr_image, axis=0)  # Añadir dimensión para batch
    sr_image = model.predict(lr_image, verbose=0)[0]  # Predecir y quitar batch
    sr_image = np.clip(sr_image, 0, 1)  # Asegurar valores entre 0 y 1
    return sr_image

# Función para mostrar imágenes
def show_results(lr_image, sr_image, hr_image=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Low Resolution")
    plt.imshow(lr_image)

    plt.subplot(1, 3, 2)
    plt.title("Super Resolution")
    plt.imshow(sr_image)

    if hr_image is not None:
        plt.subplot(1, 3, 3)
        plt.title("High Resolution")
        plt.imshow(hr_image)

    plt.show()

if __name__ == "__main__":
    try:
        # Cargar el modelo entrenado
        model = tf.keras.models.load_model("super_resolution_model.keras")
        print("Modelo cargado exitosamente.")

        # Directorio de imágenes de baja resolución (LR)
        lr_folder = r"img_prueba"  # Cambia esta ruta si es necesario

        # Cargar las imágenes de baja resolución
        lr_images, filenames = load_images(lr_folder)

        if len(lr_images) == 0:
            print("No se encontraron imágenes en la carpeta de baja calidad.")
        else:
            # Probar con la primera imagen de baja resolución
            test_lr_image = lr_images[0]
            sr_image = upscale_image(model, test_lr_image)

            # Cargar una imagen de alta resolución para comparación (opcional)
            hr_image_path = r"buena_calidad\Fortnite-2019_07_26-17_59_50_000001.jpg"  # Cambia esta ruta si es necesario
            hr_image = None
            if os.path.exists(hr_image_path):
                hr_image = load_img(hr_image_path)
                hr_image = img_to_array(hr_image) / 255.0  # Normalizar

            # Mostrar resultados
            show_results(test_lr_image, sr_image, hr_image)

            # Guardar resultado SR
            output_path = "resultado_super_resolucion.png"
            sr_image_uint8 = (sr_image * 255).astype(np.uint8)  # Convertir a formato de imagen
            cv2.imwrite(output_path, cv2.cvtColor(sr_image_uint8, cv2.COLOR_RGB2BGR))
            print(f"Imagen de super resolución guardada en: {output_path}")
    except Exception as e:
        print(f"Error: {e}")