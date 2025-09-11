import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import time

# --- FASE 0: CONFIGURACIÓN ---
# Modifica esta ruta a tu carpeta 'train'
base_dir = r'C:\DiscoLocalD\Programacion\Trabajo\Clasificacion_Carnes\dataset_final\train'
class_names = ['ALPACA', 'LLAMA']
target_per_class = 1000

print("Iniciando el proceso de replicación en ALTA CALIDAD.")
print(f"Objetivo: {target_per_class} imágenes por clase, preservando detalles.")

# --- GENERADOR CON TRANSFORMACIONES SUTILES PARA CREAR "RÉPLICAS" ---
# Los rangos son mucho más pequeños para no distorsionar la imagen.
datagen = ImageDataGenerator(
    rotation_range=10,          # Rotación muy ligera
    width_shift_range=0.05,     # Movimiento lateral mínimo
    height_shift_range=0.05,    # Movimiento vertical mínimo
    zoom_range=0.05,            # Zoom casi imperceptible
    horizontal_flip=True,       # Volteo horizontal (seguro para este caso)
    brightness_range=[0.95, 1.05], # Variación de brillo mínima
    fill_mode='reflect'         # 'reflect' suele dar mejores resultados en bordes que 'nearest'
)

# --- FASE 1: GENERACIÓN DE RÉPLICAS EN ALTA CALIDAD ---
for class_name in class_names:
    dir_path = os.path.join(base_dir, class_name)

    if not os.path.exists(dir_path):
        print(f"ADVERTENCIA: El directorio no existe, saltando: {dir_path}")
        continue

    existing_images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_existing = len(existing_images)

    print(f"\nClase: '{class_name}'")
    print(f"Imágenes originales encontradas: {num_existing}")

    num_to_generate = target_per_class - num_existing

    if num_to_generate <= 0:
        print("Objetivo ya alcanzado. Pasando a la siguiente fase.")
        continue

    print(f"Generando {num_to_generate} réplicas de alta calidad...")
    
    images_to_augment = random.choices(existing_images, k=num_to_generate)
    
    for i, image_name in enumerate(images_to_augment):
        img_path = os.path.join(dir_path, image_name)
        try:
            # --- CAMBIO CLAVE: Se carga la imagen en su resolución original completa ---
            # No se usa 'target_size', por lo que la calidad se mantiene intacta.
            img = tf.keras.utils.load_img(img_path)
            x = tf.keras.utils.img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Guardar la nueva réplica con un prefijo temporal
            j = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=dir_path,
                                      save_prefix='temp_aug_', # Usamos un prefijo temporal
                                      save_format='jpg'):
                j += 1
                if j >= 1:
                    break
        except Exception as e:
            print(f"Error procesando {image_name}: {e}")

    print(f"Generación para '{class_name}' completa.")


print("\n--- FASE 2: RENOMBRADO Y ORGANIZACIÓN FINAL DEL DIRECTORIO ---")
for class_name in class_names:
    dir_path = os.path.join(base_dir, class_name)
    if not os.path.exists(dir_path):
        continue

    all_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Renombrando {len(all_files)} archivos en la carpeta '{class_name}'...")

    for i, old_filename in enumerate(all_files):
        # El nuevo nombre será CLASE_XXXX.jpg (ej. ALPACA_0001.jpg)
        # El :04d asegura que el número siempre tenga 4 dígitos (0001, 0002, ..., 0999, 1000)
        new_filename = f"{class_name}_{i+1:04d}.jpg"
        
        old_filepath = os.path.join(dir_path, old_filename)
        new_filepath = os.path.join(dir_path, new_filename)
        
        # Renombrar el archivo
        os.rename(old_filepath, new_filepath)

    print(f"Renombrado completo para '{class_name}'.")


print("\n--- PROCESO FINALIZADO ---")
print("Verificación final del directorio:")

for class_name in class_names:
    dir_path = os.path.join(base_dir, class_name)
    if os.path.exists(dir_path):
        final_count = len([f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')])
        print(f"Conteo final en '{class_name}': {final_count} imágenes.")
        # Opcional: Listar los primeros 10 archivos para verificar el nuevo nombre
        print("Ejemplo de nuevos nombres:", sorted(os.listdir(dir_path))[:10])