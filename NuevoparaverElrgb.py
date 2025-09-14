# 1. Monta tu Drive como siempre
from google.colab import drive
drive.mount('/content/drive')

# 2. ¡EL PASO CLAVE! Descomprime tu dataset desde Drive al disco local de Colab
# Cambia la ruta a donde tengas tu archivo zip
#!unzip "/content/drive/MyDrive/Colab Notebooks/datos.zip" -d "/content/dataset/"

# 3. Apunta tu script a la nueva carpeta local súper rápida
data_dir = '/content/dataset/train' # O la ruta que corresponda dentro del zip

# --- 0. MONTAR GOOGLE DRIVE Y CONFIGURACIÓN INICIAL ---
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. CONFIGURACIÓN ---
data_dir = '/content/drive/MyDrive/Colab Notebooks/train'
img_height = 224
img_width = 224
max_epochs = 50 # Las épocas se repartirán entre las dos fases

# <-- ¡MODIFICADO! Aumentamos el tamaño del lote para un entrenamiento más estable.
batch_size = 32

# --- 2. CAPA DE AUMENTO DE DATOS ---
# Mantenemos el aumento de datos agresivo que es bueno para la generalización.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), # La voltereta vertical puede no tener sentido para animales
    layers.RandomRotation(0.2), # Rotación de hasta 20%
    layers.RandomZoom(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Reducimos un poco el desplazamiento
    layers.RandomBrightness(factor=0.2)
], name="data_augmentation")

# --- 3. CARGA Y PREPROCESAMIENTO DE DATOS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Clases encontradas:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. CONSTRUCCIÓN DEL MODELO ---
# Usamos ResNet152 como antes, pero lo preparamos para el fine-tuning
preprocess_input = tf.keras.applications.resnet.preprocess_input
base_model = tf.keras.applications.ResNet152(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
# Preparamos para fine-tuning: descongelamos algunas capas más tarde
base_model.trainable = False

# Construcción del modelo completo
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False) # La base_model se ejecuta en modo inferencia
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
# Aseguramos que la salida sea una sola neurona para clasificación binaria.
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# --- 5. ENTRENAMIENTO DEL MODELO (SOLO CABEZA) ---
print("\n--- INICIANDO ENTRENAMIENTO (SOLO CABEZA) CON EARLY STOPPING ---")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Entrenamos solo la cabeza (la capa densa superior)
history_head = model.fit(
    train_ds,
    epochs=max_epochs // 2, # La mitad de las épocas para la cabeza
    validation_data=val_ds,
    callbacks=[early_stopping]
)

# --- 6. FINE-TUNING DEL MODELO ---
print("\n--- INICIANDO FINE-TUNING CON EARLY STOPPING ---")

# Descongelamos algunas capas de la base_model para fine-tuning
base_model.trainable = True

# Es importante recompilar el modelo después de cambiar la propiedad trainable de las capas
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Tasa de aprendizaje más baja para fine-tuning
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Continuamos el entrenamiento con la tasa de aprendizaje más baja
history_fine_tune = model.fit(
    train_ds,
    epochs=max_epochs // 2, # La otra mitad de las épocas para fine-tuning
    validation_data=val_ds,
    callbacks=[early_stopping]
)

# Combinamos los historiales de entrenamiento para la visualización
acc = history_head.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history_head.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history_head.history['loss'] + history_fine_tune.history['loss']
val_loss = history_head.history['val_loss'] + history_fine_tune.history['val_loss']

# --- 7. VISUALIZACIÓN DEL HISTORIAL DE ENTRENAMIENTO ---

print("\n✅ Generando gráfico del historial de entrenamiento...")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión de Entrenamiento')
plt.plot(val_acc, label='Precisión de Validación')
plt.ylim([min(plt.ylim()), 1])
plt.title('Precisión a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.ylim([0, max(plt.ylim())])
plt.title('Pérdida a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')

plt.tight_layout()
nombre_grafico = 'historial_entrenamiento_resnet152_aug_ft.png'
plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/{nombre_grafico}')
print(f"✅ Gráfico guardado en tu Drive como '{nombre_grafico}'")

#¡NUEVO! 8. EVALUACIÓN CON MATRIZ DE CONFUSIÓN ---

print("\n матрица путаницы Generando Matriz de Confusión...")

# Extraer las etiquetas verdaderas y las predicciones del conjunto de validación
y_true = []
y_pred_probs = []

# Iteramos sobre el dataset de validación para obtener todas las etiquetas y predicciones
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds.flatten())

# Convertir a arrays de numpy
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Convertir probabilidades (ej. 0.98) a etiquetas de clase (0 o 1)
y_pred = (y_pred_probs > 0.5).astype(int)

# Calcular la matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')

plt.title('Matriz de Confusión')
plt.xticks(rotation=45)

# --- 8. GUARDADO DEL MODELO ---
nombre_archivo_modelo = 'clasificador_carne_resnet152_aug_ft.keras'
model.save(f'/content/drive/MyDrive/Colab Notebooks/{nombre_archivo_modelo}')
print(f"\n¡Nuevo modelo guardado en tu Drive como '{nombre_archivo_modelo}'! ✅")