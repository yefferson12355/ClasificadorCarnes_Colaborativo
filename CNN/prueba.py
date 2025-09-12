# =============================================================================
# === SCRIPT UNIFICADO: CLASIFICACIÓN DE CARNE DE CAMÉLIDOS (DEEP LEARNING) ===
# =============================================================================
# Este script realiza el ciclo completo de investigación:
# 1. Define y compara 4 arquitecturas de CNN (VGG16, ResNet50, MobileNetV2, EfficientNetB0).
# 2. Para cada modelo:
#    a. Construye la arquitectura con una cabeza de clasificación común.
#    b. Entrena en 2 fases (transfer learning y fine-tuning).
#    c. Guarda el modelo entrenado y su historial de aprendizaje en un gráfico.
#    d. Evalúa el modelo en un set de prueba y guarda la matriz de confusión.
# 3. Al final, realiza un análisis de interpretabilidad (Grad-CAM) sobre el mejor modelo.
# 4. Imprime una tabla resumen comparando el rendimiento de todos los modelos.

# --- Importación de Librerías ---
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm

# =============================================================================
# --- 1. CONFIGURACIÓN GLOBAL ---
# =============================================================================
# ▼▼▼ ¡VERIFICA ESTAS RUTAS! ▼▼▼
DATA_DIR_TRAIN = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\train'
DATA_DIR_TEST = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\test'
SAMPLE_IMAGE_FOR_GRADCAM = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\test\ALPACA\1.jpg' # Una imagen para el análisis final

# --- Hiperparámetros de Entrenamiento ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
CLASS_NAMES = ['ALPACA', 'LLAMA'] # Asegúrate de que coincida con el orden de las carpetas

# --- Modelos a Comparar ---
MODELS_TO_COMPARE = ['ResNet50', 'MobileNetV2', 'VGG16', 'EfficientNetB0']

# =============================================================================
# --- 2. FUNCIONES AUXILIARES ---
# =============================================================================

def create_model(model_name, num_classes=1):
    """
    Crea un modelo de Keras basado en una arquitectura pre-entrenada.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'

    # Seleccionar el modelo base
    if model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    elif model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    elif model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    elif model_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    base_model.trainable = False  # Congelar la base inicialmente

    # Capa de aumento de datos
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    # Función de preprocesamiento específica del modelo
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input if model_name == 'MobileNetV2' else \
                     tf.keras.applications.resnet.preprocess_input if model_name == 'ResNet50' else \
                     tf.keras.applications.vgg16.preprocess_input if model_name == 'VGG16' else \
                     tf.keras.applications.efficientnet.preprocess_input

    # Construir el modelo completo
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation=activation)(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    
    return model

def plot_history(history, history_fine, model_name):
    """
    Genera y guarda un gráfico del historial de entrenamiento.
    """
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Precisión de Entrenamiento')
    plt.plot(val_acc, label='Precisión de Validación')
    plt.axvline(INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Inicio de Ajuste Fino')
    plt.ylim([min(plt.ylim()), 1.01])
    plt.title(f'Precisión - {model_name}')
    plt.xlabel('Época'); plt.ylabel('Precisión')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    plt.axvline(INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Inicio de Ajuste Fino')
    plt.title(f'Pérdida - {model_name}')
    plt.xlabel('Época'); plt.ylabel('Pérdida')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'historial_entrenamiento_{model_name}.png')
    plt.close()
    print(f"✅ Gráfico de historial guardado como 'historial_entrenamiento_{model_name}.png'")

def evaluate_model(model, test_data_path, model_name, test_type="Test con Imágenes Nuevas"):
    """
    Evalúa un modelo en un directorio de prueba y genera una matriz de confusión.
    """
    print(f"\n--- Evaluando {model_name} en '{test_type}' ---")
    y_true = []
    y_pred_probs = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_data_path, class_name)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                pred_prob = model.predict(img_array, verbose=0)[0][0]
                y_pred_probs.append(pred_prob)
                y_true.append(CLASS_NAMES.index(class_name))

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"🎯 Exactitud (Accuracy): {accuracy:.2%}")
    print(f"📊 Precisión: {precision:.2%}, Recuperación (Recall): {recall:.2%}, Puntuación F1: {f1:.2%}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicción del Modelo'); plt.ylabel('Etiqueta Real')
    plt.title(f'Matriz de Confusión - {model_name} ({test_type})')
    
    plt.savefig(f'matriz_confusion_{model_name}_{test_type.replace(" ", "_")}.png')
    plt.close()
    print(f"✅ Matriz de confusión guardada como 'matriz_confusion_{model_name}_{test_type.replace(' ', '_')}.png'")
    
    return accuracy, precision, recall, f1

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def visualize_gradcam(model, image_path, model_name):
    print(f"\n--- Visualizando Grad-CAM para {model_name} ---")
    img_array = get_img_array(image_path, size=(IMG_HEIGHT, IMG_WIDTH))

    last_conv_layer_name = None
    for layer in reversed(model.get_layer(model_name).layers): # Buscar dentro del submodelo
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    
    if not last_conv_layer_name:
        print("❌ No se pudo encontrar la última capa convolucional.")
        return

    full_conv_name = f"{model_name}/{last_conv_layer_name}"
    
    heatmap = make_gradcam_heatmap(img_array, model, full_conv_name)
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.6 + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(f"gradcam_resultado_{model_name}.png")
    print(f"✅ Grad-CAM guardado como 'gradcam_resultado_{model_name}.png'")

def get_img_array(img_path, size):
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# =============================================================================
# --- 3. EJECUCIÓN PRINCIPAL DEL PROYECTO ---
# =============================================================================
if __name__ == "__main__":
    
    # --- Cargar y preparar los datos una sola vez ---
    print("--- Cargando y preparando datasets de entrenamiento y validación ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    
    # Optimizar datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    results = []

    # --- Bucle de entrenamiento y evaluación para cada modelo ---
    for model_name in MODELS_TO_COMPARE:
        print(f"\n\n{'='*70}")
        print(f"🚀 PROCESANDO MODELO: {model_name}")
        print(f"{'='*70}")
        
        # --- Construcción del modelo ---
        model = create_model(model_name)
        
        # --- Etapa 1: Entrenar la cabeza ---
        print(f"\n--- {model_name}: Iniciando Etapa 1 (Entrenando cabeza) ---")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds)
        
        # --- Etapa 2: Ajuste Fino ---
        print(f"\n--- {model_name}: Iniciando Etapa 2 (Ajuste Fino) ---")
        base_model_internal = model.get_layer(model_name.lower()) # Obtener el submodelo
        base_model_internal.trainable = True
        
        # Congelar solo una parte de la base
        fine_tune_at = len(base_model_internal.layers) // 2
        for layer in base_model_internal.layers[:fine_tune_at]:
            layer.trainable = False
            
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        history_fine = model.fit(train_ds, epochs=TOTAL_EPOCHS, 
                                 initial_epoch=history.epoch[-1], validation_data=val_ds)
        
        # --- Guardar modelo y gráficos ---
        model_filename = f'clasificador_carne_{model_name}.keras'
        model.save(model_filename)
        print(f"\n✅ Modelo guardado como '{model_filename}'")
        plot_history(history, history_fine, model_name)
        
        # --- Evaluar el modelo ---
        accuracy, precision, recall, f1 = evaluate_model(model, DATA_DIR_TEST, model_name)
        
        # Guardar resultados para el resumen final
        model_size = os.path.getsize(model_filename) / (1024 * 1024) # en MB
        results.append({
            'Modelo': model_name,
            'Exactitud (Accuracy)': f"{accuracy:.2%}",
            'Precisión': f"{precision:.2%}",
            'Recuperación (Recall)': f"{recall:.2%}",
            'Puntuación F1': f"{f1:.2%}",
            'Tamaño (MB)': f"{model_size:.2f}"
        })

    # --- Encontrar el mejor modelo y visualizar Grad-CAM ---
    best_model_result = max(results, key=lambda x: float(x['Exactitud (Accuracy)'].strip('%')))
    best_model_name = best_model_result['Modelo']
    
    print(f"\n\n{'='*70}")
    print(f"🏆 MEJOR MODELO DETERMINADO: {best_model_name}")
    print(f"{'='*70}")
    
    best_model = tf.keras.models.load_model(f'clasificador_carne_{best_model_name}.keras')
    visualize_gradcam(best_model, SAMPLE_IMAGE_FOR_GRADCAM, best_model_name)

    # --- Imprimir tabla de resumen final ---
    print(f"\n\n{'='*70}")
    print("📊 TABLA COMPARATIVA DE RESULTADOS FINALES 📊")
    print(f"{'='*70}")
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))