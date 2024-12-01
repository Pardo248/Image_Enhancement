import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
import os
import cv2
import numpy as np

# Generador para cargar imágenes desde el disco
class ImageSequence(Sequence):
    def __init__(self, hr_folder, lr_folder, batch_size, target_shape=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.batch_size = batch_size
        self.target_shape = target_shape
        
        # Obtener listas de imágenes
        self.hr_images = sorted([os.path.join(hr_folder, f) for f in os.listdir(hr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_images = sorted([os.path.join(lr_folder, f) for f in os.listdir(lr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.hr_images) != len(self.lr_images):
            raise ValueError("El número de imágenes en las carpetas HR y LR no coincide.")
        
        self.num_samples = len(self.hr_images)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        # Obtener índices para este lote
        batch_hr = self.hr_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_lr = self.lr_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        hr_images, lr_images = [], []
        
        for hr_path, lr_path in zip(batch_hr, batch_lr):
            # Leer imágenes HR y LR
            hr_img = cv2.imread(hr_path)
            lr_img = cv2.imread(lr_path)
            
            if self.target_shape:
                hr_img = cv2.resize(hr_img, (self.target_shape[1], self.target_shape[0]), interpolation=cv2.INTER_CUBIC)
                lr_img = cv2.resize(lr_img, (self.target_shape[1], self.target_shape[0]), interpolation=cv2.INTER_CUBIC)

            # Normalizar a valores entre 0 y 1
            hr_images.append(hr_img / 255.0)
            lr_images.append(lr_img / 255.0)
        
        return np.array(lr_images), np.array(hr_images)

# Modelo de super-resolución basado en CNN
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
def train_model(model, train_gen, epochs=10):
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse', metrics=['accuracy'])

    # Guardar el mejor modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Entrenamiento del modelo
    history = model.fit(
        train_gen,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    return history

if __name__ == "__main__":
    # Directorios de las carpetas
    hr_folder = r"buena_calidad"
    lr_folder = r"mala_calidad"

    # Definir tamaño objetivo (opcional) y tamaño de lote
    target_shape = (540, 960)  # Ajustar según las necesidades
    batch_size = 8

    # Crear el generador de imágenes
    train_gen = ImageSequence(hr_folder, lr_folder, batch_size=batch_size, target_shape=target_shape)

    # Construir y entrenar el modelo
    model = build_model()
    history = train_model(model, train_gen, epochs=10)

    # Guardar el modelo
    model.save("super_resolution_model.keras")
