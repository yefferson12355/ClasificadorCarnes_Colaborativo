# --- Importación de Librerías ---
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns # Para una matriz de confusión más bonita
from sklearn.metrics import confusion_matrix # Para calcular la matriz

# =============================================================================
# --- 1. CONFIGURACIÓN DE RUTAS ---
# =============================================================================
#RUTA_DEL_MODELO = r'clasificador_carne_resnet50_v4_rgb.keras' # Asegúrate de usar el nuevo modelo entrenado
RUTA_DEL_MODELO = r'clasificador_carne_resnet50_v4_solo_cabeza.keras'
RUTA_TEST_NUEVO = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\CNN\test'
RUTA_TEST_APRENDIDO = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\CNN\img_ya_aprendidas_duplicadas'

img_height = 224
img_width = 224

# =============================================================================
# --- 2. FUNCIÓN DE PREPROCESAMIENTO (AHORA USA IMÁGENES RGB) ---
# =============================================================================
# NOTA: Como el último entrenamiento fue con imágenes RGB, el preprocesamiento
#       CLAHE+Sobel ya no es necesario aquí. La función de evaluación ahora
#       trabajará directamente con el preprocesamiento de ResNet.

# =============================================================================
# --- 3. FUNCIÓN PRINCIPAL DE EVALUACIÓN (ACTUALIZADA) ---
# =============================================================================
def evaluar_directorio(model, data_path, class_names, test_name):
    print(f"\n\n{'='*60}")
    print(f" 🚀 INICIANDO EVALUACIÓN: {test_name.upper()} ")
    print(f"{'='*60}")

    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró el directorio de evaluación en '{data_path}'")
        return

    # Listas para guardar las etiquetas reales y las predicciones
    y_true = []
    y_pred = []

    for clase_real_str in class_names:
        folder_path = os.path.join(data_path, clase_real_str)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n--- Evaluando imágenes de la clase: {clase_real_str} ---")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                
                img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                
                # Preprocesamiento de ResNet directamente
                img_resnet_processed = tf.keras.applications.resnet.preprocess_input(img_array)
                img_batch = np.expand_dims(img_resnet_processed, axis=0)
                
                prediccion = model.predict(img_batch, verbose=0)
                score = prediccion[0][0]
                clase_predicha_str = class_names[1] if score >= 0.5 else class_names[0]
                
                # Guardar resultados para la matriz de confusión
                y_true.append(clase_real_str)
                y_pred.append(clase_predicha_str)

    # --- Resumen en texto ---
    total_imagenes = len(y_true)
    if total_imagenes == 0:
        print("No se encontraron imágenes para evaluar.")
        return
        
    correctas = np.sum(np.array(y_true) == np.array(y_pred))
    accuracy = (correctas / total_imagenes) * 100
    
    print(f"\n--- RESUMEN DE LA EVALUACIÓN: {test_name.upper()} ---")
    print(f"Total de imágenes: {total_imagenes}, Correctas: {correctas}, Incorrectas: {total_imagenes - correctas}")
    print(f"🎯 Precisión (Accuracy): {accuracy:.2f}%")

    # --- Generar y guardar la Matriz de Confusión ---
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción del Modelo', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.title(f'Matriz de Confusión ({test_name})', fontsize=15)
    
    nombre_matriz = f'matriz_confusion_{test_name.lower().replace(" ", "_")}.png'
    plt.savefig(nombre_matriz)
    print(f"✅ Matriz de Confusión guardada como '{nombre_matriz}'")
    plt.close() # Cerrar la figura para liberar memoria

# =============================================================================
# --- 4. EJECUCIÓN PRINCIPAL ---
# =============================================================================
if not os.path.exists(RUTA_DEL_MODELO):
    print(f"❌ Error CRÍTICO: No se encontró el modelo en '{RUTA_DEL_MODELO}'")
else:
    try:
        print("✅ Cargando modelo (esto puede tardar un momento)...")
        # Asegúrate de que el preprocesamiento personalizado no sea necesario si no se usó en el entrenamiento
        model = tf.keras.models.load_model(RUTA_DEL_MODELO, compile=False)
        model.compile(metrics=['accuracy']) # Re-compilar es una buena práctica después de cargar
        
        class_names = ['ALPACA', 'LLAMA']
        
        evaluar_directorio(model, RUTA_TEST_NUEVO, class_names, "Test con Imágenes Nuevas")
        evaluar_directorio(model, RUTA_TEST_APRENDIDO, class_names, "Test con Imágenes de Entrenamiento")

    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")