import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# --- 1. CONFIGURACIÓN ---

data_dir = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\CNN\train'
img_height = 224
img_width = 224
batch_size = 32 # 
# <-- ¡MODIFICADO! Aumentamos las épocas máximas para que EarlyStopping tenga margen para trabajar.
max_epochs = 50 # 100


# --- 2. CAPA DE AUMENTO DE DATOS ---

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])


# --- 3. CARGA Y PREPROCESAMIENTO DE DATOS ---

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


# --- 4. CONSTRUCCIÓN DEL MODELO ---

base_model = tf.keras.applications.ResNet50(  #ResNet50  V3
    input_shape=(img_height, img_width, 3),   #ResNet151 y  ResNet101
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

# --- 5. ETAPA ÚNICA DE ENTRENAMIENTO (SIN AJUSTE FINO) ---

print("\n--- INICIANDO ENTRENAMIENTO (SOLO CABEZA) CON EARLY STOPPING ---")

# Definimos EarlyStopping para que monitoree la pérdida de validación
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Le damos un poco más de paciencia (10 épocas) por si hay fluctuaciones
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Entrenamos en una sola etapa, dejando que EarlyStopping haga su trabajo
history = model.fit(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
    callbacks=[early_stopping] # Aplicamos el callback aquí
)


# --- 6. VISUALIZACIÓN DEL HISTORIAL DE ENTRENAMIENTO ---

print("\n✅ Generando gráfico del historial de entrenamiento...")

# <-- ¡MODIFICADO! La lógica se simplifica ya que solo hay un 'history'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))

# Subplot para la Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión de Entrenamiento')
plt.plot(val_acc, label='Precisión de Validación')
plt.ylim([min(plt.ylim()), 1])
plt.title('Precisión a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(loc='lower right')

# Subplot para la Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.ylim([0, max(plt.ylim())])
plt.title('Pérdida a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')

plt.tight_layout()
nombre_grafico = 'historial_entrenamiento_v4_solo_cabeza.png'
plt.savefig(nombre_grafico)
print(f"✅ Gráfico guardado como '{nombre_grafico}'")

# --- 7. GUARDADO DEL MODELO ---
# <-- ¡MODIFICADO! Nuevo nombre para el modelo que refleja la nueva estrategia.
nombre_archivo_modelo = 'clasificador_carne_resnet50_v4_solo_cabeza.keras'
model.save(nombre_archivo_modelo)
print(f"\n¡Nuevo modelo guardado como '{nombre_archivo_modelo}'! ✅")