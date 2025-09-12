# --- Importación de Librerías ---
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# =============================================================================
# --- 1. CONFIGURACIÓN DE RUTAS ---
# =============================================================================
# ▼▼▼ ¡VERIFICA ESTAS TRES RUTAS! ▼▼▼
RUTA_DEL_MODELO = r'clasificador_carne_resnet50_con_procesamiento_integrado.keras'

# Ruta para las imágenes de TESTEO REAL (imágenes nuevas o "malas")
RUTA_TEST_NUEVO = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\test'

# Ruta para las imágenes YA VISTAS en el entrenamiento
RUTA_TEST_APRENDIDO = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\img_ya_aprendidas_duplicadas'

# Dimensiones de la imagen
img_height = 224
img_width = 224

# =============================================================================
# --- 2. FUNCIÓN DE PREPROCESAMIENTO (Sin cambios) ---
# =============================================================================
def apply_clahe_sobel(image_np):
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_img)
    sobel_x = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=5)
    processed_img = cv2.convertScaleAbs(sobel_x)
    img_3_channels = cv2.merge([processed_img, processed_img, processed_img])
    return img_3_channels

# =============================================================================
# --- 3. FUNCIÓN PRINCIPAL DE EVALUACIÓN ---
# =============================================================================
def evaluar_directorio(model, data_path, class_names, test_name):
    """
    Evalúa todas las imágenes en un directorio y genera un resumen y un gráfico.
    """
    print(f"\n\n{'='*60}")
    print(f" 🚀 INICIANDO EVALUACIÓN: {test_name.upper()} ")
    print(f"{'='*60}")

    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró el directorio de evaluación en '{data_path}'")
        return

    resultados = {name: {'correctas': 0, 'incorrectas': 0} for name in class_names}

    for clase_real_str in class_names:
        folder_path = os.path.join(data_path, clase_real_str)
        if not os.path.isdir(folder_path):
            print(f"  -> Advertencia: No se encontró la carpeta para la clase {clase_real_str}")
            continue

        print(f"\n--- Evaluando imágenes de la clase: {clase_real_str} ---")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                
                img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_processed = apply_clahe_sobel(img_array)
                img_resnet_processed = tf.keras.applications.resnet.preprocess_input(img_processed)
                img_batch = np.expand_dims(img_resnet_processed, axis=0)
                
                prediccion = model.predict(img_batch, verbose=0)
                score = prediccion[0][0]
                clase_predicha_str = class_names[1] if score >= 0.5 else class_names[0]
                
                if clase_predicha_str == clase_real_str:
                    resultados[clase_real_str]['correctas'] += 1
                else:
                    resultados[clase_real_str]['incorrectas'] += 1

    # --- Resumen en texto ---
    total_correctas = sum(res['correctas'] for res in resultados.values())
    total_incorrectas = sum(res['incorrectas'] for res in resultados.values())
    total_imagenes = total_correctas + total_incorrectas

    print(f"\n--- RESUMEN DE LA EVALUACIÓN: {test_name.upper()} ---")
    if total_imagenes > 0:
        accuracy = (total_correctas / total_imagenes) * 100
        print(f"Total de imágenes: {total_imagenes}, Correctas: {total_correctas}, Incorrectas: {total_incorrectas}")
        print(f"🎯 Precisión (Accuracy): {accuracy:.2f}%")
    else:
        print("No se encontraron imágenes para evaluar.")
        return

    # --- Generar gráfico ---
    labels = class_names
    correctas = [resultados[name]['correctas'] for name in class_names]
    incorrectas = [resultados[name]['incorrectas'] for name in class_names]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width/2, correctas, width, label='Predicción Correcta', color='mediumseagreen')
    rects2 = ax.bar(x + width/2, incorrectas, width, label='Predicción Incorrecta', color='indianred')

    ax.set_ylabel('Cantidad de Imágenes')
    ax.set_title(f'Resultados de Predicción ({test_name})', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    nombre_grafico = f'resultados_{test_name.lower().replace(" ", "_")}.png'
    plt.savefig(nombre_grafico)
    print(f"✅ Gráfico guardado como '{nombre_grafico}'")

# =============================================================================
# --- 4. EJECUCIÓN PRINCIPAL ---
# =============================================================================
if not os.path.exists(RUTA_DEL_MODELO):
    print(f"❌ Error CRÍTICO: No se encontró el modelo en '{RUTA_DEL_MODELO}'")
else:
    try:
        print("✅ Cargando modelo (esto puede tardar un momento)...")
        model = tf.keras.models.load_model(RUTA_DEL_MODELO)
        class_names = ['ALPACA', 'LLAMA']
        
        # --- Ejecutar la primera evaluación (datos de test) ---
        evaluar_directorio(model, RUTA_TEST_NUEVO, class_names, "Test con Imágenes Nuevas")
        
        # --- Ejecutar la segunda evaluación (datos ya aprendidos) ---
        evaluar_directorio(model, RUTA_TEST_APRENDIDO, class_names, "Test con Imágenes de Entrenamiento")

    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")