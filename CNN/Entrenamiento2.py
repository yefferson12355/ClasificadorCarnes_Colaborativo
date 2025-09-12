import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt # Se importa para los gráficos
import cv2
import numpy as np
import os

# =============================================================================
# --- 1. CONFIGURACIÓN ---
# =============================================================================
data_dir = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\train'
img_height = 224
img_width = 224
batch_size = 32
initial_epochs = 15
fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs

# =============================================================================
# --- 2. CAPA DE AUMENTO DE DATOS ---
# =============================================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# =============================================================================
# --- 3. CARGA Y PREPROCESAMIENTO DE DATOS ---
# =============================================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, subset="training", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, subset="validation", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size
)

class_names = train_ds.class_names
print("Clases encontradas:", class_names)
AUTOTUNE = tf.data.AUTOTUNE

preprocess_input = tf.keras.applications.resnet.preprocess_input
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =============================================================================
# --- 4. CONSTRUCCIÓN DEL MODELO ---
# =============================================================================
base_model = tf.keras.applications.ResNet50(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# =============================================================================
# --- 5. ETAPAS DE ENTRENAMIENTO ---
# =============================================================================
print("\n--- INICIANDO ETAPA 1: Entrenando solo la cabeza ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=val_ds
)

print("\n--- INICIANDO ETAPA 2: Ajuste fino (Fine-Tuning) ---")
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_ds
)

# =============================================================================
# --- 6. VISUALIZACIÓN DEL HISTORIAL DE ENTRENAMIENTO (NUEVO) ---
# =============================================================================
print("\n✅ Generando gráfico del historial de entrenamiento...")

# Combinar los historiales de las dos etapas
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Crear la figura con dos subplots
plt.figure(figsize=(12, 6))

# Subplot para la Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión de Entrenamiento')
plt.plot(val_acc, label='Precisión de Validación')
plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Inicio de Ajuste Fino')
plt.ylim([min(plt.ylim()), 1])
plt.title('Precisión a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(loc='lower right')

# Subplot para la Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Inicio de Ajuste Fino')
plt.ylim([0, max(plt.ylim())])
plt.title('Pérdida a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')

# Guardar la figura
plt.tight_layout()
nombre_grafico = 'historial_entrenamiento.png'
plt.savefig(nombre_grafico)
print(f"✅ Gráfico guardado como '{nombre_grafico}'")


# =============================================================================
# --- 7. GUARDADO DEL MODELO ---
# =============================================================================
nombre_archivo_modelo = 'clasificador_carne_resnet50_v2_rgb.keras'
model.save(nombre_archivo_modelo)
print(f"\n¡Nuevo modelo guardado como '{nombre_archivo_modelo}'! ✅")