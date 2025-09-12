import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


# En avanze


# ============================
# 1. Cargar modelo entrenado
# ============================
print("✅ Cargando modelo...")
modelo_path = r"C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\clasificador_carne_resnet50_v2_rgb.keras"
model = tf.keras.models.load_model(modelo_path)

# ============================
# 2. Función para cargar imagen
# ============================
def cargar_imagen(ruta, img_height=224, img_width=224):
    img = tf.keras.utils.load_img(ruta, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch de 1
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    return img_array, img

# ============================
# 3. Localizar última capa conv
# ============================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:  # salida con 4 dimensiones (feature maps)
            return layer.name
    raise ValueError("❌ No se encontró una capa convolucional en el modelo.")

ultima_conv = get_last_conv_layer(model)
print(f"🔎 Última capa convolucional usada para Grad-CAM: {ultima_conv}")

# ============================
# 4. Grad-CAM
# ============================
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])  # clase con mayor probabilidad
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap

# ============================
# 5. Visualizar patrones (filtros)
# ============================
def visualizar_filtros(model, layer_name, n_filtros=6):
    layer = model.get_layer(layer_name)
    pesos, _ = layer.get_weights()

    # Normalizar filtros a 0-1
    filtros = (pesos - pesos.min()) / (pesos.max() - pesos.min())

    fig, axes = plt.subplots(1, n_filtros, figsize=(20, 5))
    for i in range(n_filtros):
        f = filtros[:, :, :, i]
        f_img = (f - f.min()) / (f.max() - f.min())
        axes[i].imshow(f_img[:, :, 0], cmap="viridis")
        axes[i].axis("off")
    plt.suptitle(f"Filtros aprendidos en la capa {layer_name}", fontsize=16)
    plt.show()

# ============================
# 6. Probar con una imagen
# ============================
ruta_img = r"C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\test\ALPACA\1.jpg"

img_array, img_original = cargar_imagen(ruta_img)

# Predicción normal
pred = model.predict(img_array)
print("🔮 Predicción del modelo:", pred)

# Grad-CAM
heatmap = grad_cam(model, img_array, ultima_conv)

# Superponer el heatmap en la imagen original
heatmap = cv2.resize(heatmap, (img_original.size[0], img_original.size[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(np.array(img_original), 0.6, heatmap, 0.4, 0)

# Mostrar resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("Grad-CAM (qué miró el modelo)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================
# 7. Ver filtros aprendidos
# ============================
visualizar_filtros(model, ultima_conv, n_filtros=6)
