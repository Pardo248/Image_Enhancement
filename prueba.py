import pyautogui
import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array

# Definir el tamaño de la ventana (puedes ajustarlo)
ancho_ventana = 960
alto_ventana = 540

# Definir la región de la pantalla que se va a capturar (puedes ajustarla)
x, y, w, h = 100, 0, ancho_ventana, alto_ventana  # Captura desde la esquina superior izquierda

# Función para aumentar la resolución de una imagen
def upscale_image(model, lr_image):
    # Redimensionar la imagen a la misma resolución que se utilizó durante el entrenamiento
    target_shape = (960, 540)  # El tamaño de las imágenes que se usaron para entrenar el modelo
    lr_image = cv2.resize(lr_image, target_shape, interpolation=cv2.INTER_CUBIC)

    # Normalizar la imagen a [0, 1]
    lr_image = lr_image / 255.0

    # Añadir dimensión para el batch (esto es necesario para que funcione con el modelo)
    lr_image = np.expand_dims(lr_image, axis=0)

    # Realizar la predicción de super-resolución
    sr_image = model.predict(lr_image, verbose=0)[0]

    # Asegurar que los valores estén entre 0 y 1
    sr_image = np.clip(sr_image, 0, 1)

    # Convertir la imagen a formato de 8 bits para mostrarla
    sr_image = (sr_image * 255).astype(np.uint8)

    return sr_image

# Función para capturar y mostrar la pantalla en una pequeña ventana
def capturar_pantalla(model):
    while True:
        # Captura de la pantalla en la región definida
        screenshot = pyautogui.screenshot(region=(x, y, w, h))

        # Convertir la imagen a un formato que OpenCV puede manejar
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Aplicar el modelo de super-resolución
        screenshot_bgr = upscale_image(model, screenshot_bgr)

        # Mostrar la imagen en una ventana pequeña
        cv2.imshow("Pantalla en Ventana Pequeña", screenshot_bgr)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Cargar el modelo previamente entrenado
        model = tf.keras.models.load_model("super_resolution_model.keras")
        print("Modelo cargado exitosamente.")

        # Llamar a la función para capturar la pantalla y mostrarla
        capturar_pantalla(model)

    except Exception as e:
        print(f"Error: {e}")
