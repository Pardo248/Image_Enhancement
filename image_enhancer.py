import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint

# Ajustar dimensiones de LR para que coincidan con HR
def upscale_images(lr_images, target_shape):
    scaled_images = []
    for img in lr_images:
        scaled_img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        scaled_images.append(scaled_img)
    return np.array(scaled_images)

# Función para cargar imágenes desde las carpetas
def load_images(hr_folder, lr_folder):
    hr_images = []
    lr_images = []

    # Cargar imágenes de alta calidad (HR)
    for filename in os.listdir(hr_folder):
        file_path = os.path.join(hr_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_hr = load_img(file_path)  # Cargar imagen en formato PIL
            img_hr = img_to_array(img_hr) / 255.0  # Convertir a array y normalizar
            hr_images.append(img_hr)

    # Cargar imágenes de baja calidad (LR)
    for filename in os.listdir(lr_folder):
        file_path = os.path.join(lr_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_lr = load_img(file_path)
            img_lr = img_to_array(img_lr) / 255.0
            lr_images.append(img_lr)

    return np.array(lr_images), np.array(hr_images)

# Modelo de super-resolución simple basado en CNN
def build_model():
    model = Sequential()

    # Primera capa de convolución
    model.add(Conv2D(64, (9, 9), padding='same', input_shape=(None, None, 3)))
    model.add(Activation('relu'))

    # Capas intermedias
    for _ in range(4):
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    # Última capa para reconstrucción
    model.add(Conv2D(3, (9, 9), padding='same'))
    model.add(Activation('sigmoid'))  # Para limitar los valores entre 0 y 1

    return model

# Función para entrenar el modelo
def train_model(model, lr_images, hr_images, epochs=100, batch_size=8):
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse', metrics=['accuracy'])

    # Guardar el mejor modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    # Entrenamiento del modelo
    history = model.fit(
        lr_images, hr_images,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    return history

if __name__ == "__main__":
    # Directorios de las carpetas
    hr_folder = r"D:\machin_proyecto\buena_calidad"
    lr_folder = r"D:\machin_proyecto\mala_calidad"

    # Cargar imágenes
    lr_images, hr_images = load_images(hr_folder, lr_folder)

    # Escalar las imágenes LR a las dimensiones de HR
    lr_images = upscale_images(lr_images, hr_images.shape[1:3])  # Escalar LR a HR

    print(f"Imágenes LR escaladas: {lr_images.shape}")
    print(f"Imágenes HR: {hr_images.shape}")

    # Construir y entrenar el modelo
    model = build_model()
    history = train_model(model, lr_images, hr_images, epochs=50, batch_size=16)

    # Guardar el modelo
    model.save("super_resolution_model.h5")
