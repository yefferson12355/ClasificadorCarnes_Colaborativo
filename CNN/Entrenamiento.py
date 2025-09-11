# --- Importación de Librerías ---
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2  # Se añade OpenCV para el procesamiento
import numpy as np # Se añade NumPy para manejar los datos de imagen

# =============================================================================
# --- 1. CONFIGURACIÓN ---
# =============================================================================

data_dir = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\train' 


img_height = 224
img_width = 224
batch_size = 32
initial_epochs = 10     # Épocas para la primera etapa
fine_tune_epochs = 10   # Épocas para la segunda etapa (ajuste fino)
total_epochs = initial_epochs + fine_tune_epochs

# =============================================================================
# --- 2. FUNCIÓN DE PREPROCESAMIENTO AVANZADO (CLAHE + SOBEL) ---
# =============================================================================

def apply_clahe_sobel(image_np):
    """
    Aplica el pipeline de CLAHE + Sobel a una imagen.
    Esta función está diseñada para ser llamada desde TensorFlow.

    Args:
        image_np (numpy.ndarray): Una imagen en formato NumPy.

    Returns:
        numpy.ndarray: La imagen procesada, convertida de nuevo a 3 canales
                       para ser compatible con ResNet50.
    """
    # 1. Asegurarse de que la imagen sea de 8 bits y convertir a escala de grises.
    # TensorFlow puede pasar las imágenes como float32, OpenCV necesita uint8.
    if image_np.dtype != np.uint8:
        image_np = tf.cast(image_np, tf.uint8).numpy()

    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 2. Aplicar CLAHE para mejorar el contraste local.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_img)
    
    # 3. Aplicar filtro Sobel para resaltar fibras horizontales.
    sobel_x = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=5)
    processed_img = cv2.convertScaleAbs(sobel_x)
    
    # 4. CRÍTICO: Convertir la imagen de 1 canal (escala de grises) de nuevo a 3 canales.
    # ResNet50 requiere una entrada de 3 canales. Duplicamos el resultado en cada canal.
    img_3_channels = cv2.merge([processed_img, processed_img, processed_img])
    
    return img_3_channels

def tf_process_image(image, label):
    """
    Función "puente" que envuelve la lógica de OpenCV para usarla en un
    pipeline de tf.data.
    """
    # tf.numpy_function permite ejecutar código Python/NumPy/OpenCV puro.
    [processed_image,] = tf.numpy_function(apply_clahe_sobel, [image], [tf.uint8])
    
    # Es necesario reestablecer la forma del tensor, ya que tf.numpy_function la pierde.
    processed_image.set_shape([img_height, img_width, 3])
    
    return processed_image, label

# =============================================================================
# --- 3. CARGA Y PREPROCESAMIENTO DE DATOS (CORREGIDO) ---
# =============================================================================

# 1. Cargar los datos SIN BATCHING INICIALMENTE.
#    Cambiamos batch_size a None para que el dataset cargue imágenes una por una.
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, subset="training", seed=123,
    image_size=(img_height, img_width),
    batch_size=None  # <-- CAMBIO CLAVE: Cargar imágenes individualmente
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, subset="validation", seed=123,
    image_size=(img_height, img_width),
    batch_size=None  # <-- CAMBIO CLAVE: Cargar imágenes individualmente
)

class_names = train_ds.class_names
print("Clases encontradas:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

print("\nConstruyendo pipeline de datos con preprocesamiento integrado...")

# 2. APLICAR NUESTRO PROCESAMIENTO a cada imagen individualmente.
train_ds = train_ds.map(tf_process_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(tf_process_image, num_parallel_calls=AUTOTUNE)

# 3. Optimizar con cache ANTES de hacer batching para mayor eficiencia.
train_ds = train_ds.cache()
val_ds = val_ds.cache()

# 4. AHORA SÍ, AGRUPAR EN LOTES las imágenes ya procesadas.
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# 5. Aplicar el preprocesamiento de ResNet y optimizar la carga final.
preprocess_input = tf.keras.applications.resnet.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Pipeline listo. ✅")

# =============================================================================
# --- 4. CONSTRUCCIÓN DEL MODELO  ---
# =============================================================================
base_model = tf.keras.applications.ResNet50(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# =============================================================================
# --- 5. ETAPA 1: ENTRENAR SOLO LA CABEZA (Sin Cambios) ---
# =============================================================================
print("\n--- INICIANDO ETAPA 1: Entrenando solo la cabeza de clasificación ---")
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

# =============================================================================
# --- 6. ETAPA 2: AJUSTE FINO (FINE-TUNING) (Sin Cambios) ---
# =============================================================================
print("\n--- INICIANDO ETAPA 2: Haciendo ajuste fino (Fine-Tuning) ---")
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
# --- 7. GUARDADO DEL MODELO (Sin Cambios) ---
# =============================================================================
nombre_archivo_modelo = 'clasificador_carne_resnet50_con_procesamiento_integrado.keras'
model.save(nombre_archivo_modelo)
print(f"\n¡Modelo con Ajuste Fino guardado como '{nombre_archivo_modelo}'! ✅")

# (Aquí iría el código de visualización de resultados si lo deseas)