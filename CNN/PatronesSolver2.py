import cv2
import os
import matplotlib.pyplot as plt
import random

# --- 1. CONFIGURACIÓN ---
# ▼▼▼ ¡CAMBIA ESTA RUTA! ▼▼▼
# Coloca aquí la ruta a la carpeta que contiene tus directorios 'ALPACA' y 'LLAMA'
nueva_ruta_base = r'C:\DiscoLocalD\Programacion\Trabajo\Clasificacion_Carnes\dataset_final\train' # <-- EJEMPLO

# --- Mejora: Selección automática de imágenes ---
# Elige 4 imágenes al azar de la carpeta ALPACA para la prueba.
# Esto evita errores si los nombres de archivo no existen en la nueva carpeta.
try:
    # Obtiene la lista de todas las imágenes en la carpeta ALPACA y elige 4 al azar
    lista_imagenes_alpaca = os.listdir(os.path.join(nueva_ruta_base, 'ALPACA'))
    # Asegúrate de que los nombres de archivo sean correctos (ej: ALPACA_0025.jpg)
    images_to_test = random.sample([img for img in lista_imagenes_alpaca if img.lower().endswith('.jpg')], 4)
    print(f"Imágenes seleccionadas al azar para la prueba: {images_to_test}")
except FileNotFoundError:
    print(f"Error: No se encontró el directorio {os.path.join(nueva_ruta_base, 'ALPACA')}")
    # Si falla, usamos una lista de respaldo
    images_to_test = ['ALPACA_0025.jpg', 'ALPACA_0080.jpg', 'ALPACA_0150.jpg','ALPACA_0084.jpg']


# --- 2. FUNCIÓN DE PROCESAMIENTO (Sin cambios) ---
def process_image(image_path):
    """
    Carga una imagen, la convierte a escala de grises,
    y le aplica el filtro CLAHE + Sobel para resaltar fibras.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    sobel_x = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=5)
    processed_img = cv2.convertScaleAbs(sobel_x)
    
    return img, processed_img

# --- 3. PROCESAR Y VISUALIZAR (Lógica adaptada a la nueva ruta) ---
path_alpaca = os.path.join(nueva_ruta_base, 'ALPACA')
path_llama = os.path.join(nueva_ruta_base, 'LLAMA')

folders_to_process = {
    "ALPACA": path_alpaca,
    "LLAMA": path_llama
}

fig, axes = plt.subplots(len(images_to_test), 4, figsize=(16, 12))
fig.suptitle('Análisis de Textura: Original vs. Procesada (CLAHE + Sobel)', fontsize=20)

for col_offset, (class_name, folder_path) in enumerate(folders_to_process.items()):
    for i, img_name in enumerate(images_to_test):
        # Para LLAMA, intentamos encontrar el archivo con el mismo número
        if class_name == 'LLAMA':
            # Asume que los archivos se llaman LLAMA_XXXX.jpg
            try:
                numero = img_name.split('_')[1]
                img_name = f'LLAMA_{numero}'
            except IndexError:
                # Si el nombre no tiene el formato esperado, se salta
                continue
        
        image_path = os.path.join(folder_path, img_name)

        if not os.path.exists(image_path):
            print(f"Advertencia: No se encontró el archivo {image_path}")
            # Rellenar el espacio con un cuadro negro para no romper el layout
            ax_orig = axes[i, col_offset * 2]
            ax_orig.imshow(np.zeros((100,100), dtype=np.uint8), cmap='gray')
            ax_orig.set_title(f'No encontrado: {img_name}')
            ax_orig.axis('off')

            ax_proc = axes[i, col_offset * 2 + 1]
            ax_proc.imshow(np.zeros((100,100), dtype=np.uint8), cmap='gray')
            ax_proc.set_title(f'No encontrado: {img_name}')
            ax_proc.axis('off')
            continue

        original, procesada = process_image(image_path)
        
        if original is not None:
            ax_orig = axes[i, col_offset * 2]
            ax_orig.imshow(original, cmap='gray')
            ax_orig.set_title(f'{class_name} - {img_name} (Original)')
            ax_orig.axis('off')
            
            ax_proc = axes[i, col_offset * 2 + 1]
            ax_proc.imshow(procesada, cmap='gray')
            ax_proc.set_title(f'{class_name} - {img_name} (Procesada)')
            ax_proc.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Visualización completada.")