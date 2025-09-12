from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras import layers, models

# Parámetros
img_height, img_width = 224, 224

# Aumento de datos
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")

# Base preentrenada
base_model = tf.keras.applications.ResNet50(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Construcción del modelo
inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="input_image")
x = data_augmentation(inputs)
x = tf.keras.applications.resnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
model = tf.keras.Model(inputs, outputs, name="ResNet50_TransferLearning")

# Generar diagrama del modelo en PNG
plot_model(
    model,
    to_file="modelo_red_neuronal.png",
    show_shapes=True,        # Muestra dimensiones de entrada/salida
    show_layer_names=True,   # Muestra nombres de capas
    expand_nested=True,
    dpi=120
)

print("✅ Imagen del modelo generada como 'modelo_red_neuronal.png'")
