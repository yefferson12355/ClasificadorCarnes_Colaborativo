import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
data_dir = r'C:\DiscoLocalD\Programacion\Trabajo\Clasificacion_Carnes\datos_procesados_para_red'
img_height = 224
img_width = 224
batch_size = 32
initial_epochs = 10 # Épocas para la primera etapa
fine_tune_epochs = 10 # Épocas para la segunda etapa (ajuste fino)
total_epochs = initial_epochs + fine_tune_epochs

# --- 2. CARGA Y PREPROCESAMIENTO DE DATOS ---
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

# Preprocesamiento específico para ResNet50
preprocess_input = tf.keras.applications.resnet.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Optimización de la carga de datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. CONSTRUCCIÓN DEL MODELO ---
# Usamos ResNet50 como nuevo modelo base
base_model = tf.keras.applications.ResNet50(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Congelamos la base para la primera etapa
base_model.trainable = False

# Creamos la cabeza de clasificación
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# --- 4. ETAPA 1: ENTRENAR SOLO LA CABEZA ---
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

# --- 5. ETAPA 2: AJUSTE FINO (FINE-TUNING) ---
print("\n--- INICIANDO ETAPA 2: Haciendo ajuste fino (Fine-Tuning) ---")

# Descongelamos la base para que sus pesos puedan ser actualizados
base_model.trainable = True

# Vamos a congelar solo las primeras capas y dejar que las últimas aprendan
fine_tune_at = 100  # Descongelar desde la capa 100 en adelante
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compilamos el modelo con una tasa de aprendizaje MUY BAJA. Esto es CRÍTICO.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Continuamos el entrenamiento
history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Continuar desde donde nos quedamos
    validation_data=val_ds
)

# --- 6. VISUALIZACIÓN Y GUARDADO ---
# (El código de visualización y guardado es el mismo, pero ahora deberías
# combinar 'history' y 'history_fine' para un gráfico completo)
nombre_archivo_modelo = 'clasificador_carne_resnet50_finetuned.keras'
model.save(nombre_archivo_modelo)
print(f"\n¡Modelo con Ajuste Fino guardado como '{nombre_archivo_modelo}'! ✅")