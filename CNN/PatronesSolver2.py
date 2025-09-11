
# --- Importación de Librerías ---
import cv2  # OpenCV para el procesamiento de imágenes.
import os  # Para interactuar con el sistema operativo (rutas, archivos).
import matplotlib.pyplot as plt  # Para crear las visualizaciones y gráficos.
import random  # Para seleccionar imágenes de muestra de forma aleatoria.
import numpy as np # Necesario para crear imágenes negras si un archivo no se encuentra.

# =============================================================================
# --- SECCIÓN 1: CONFIGURACIÓN DEL SCRIPT ---
# =============================================================================
nueva_ruta_base = r'C:\DiscoLocalD\Programacion\Trabajo\Nueva carpeta\train'


try:
  
    lista_imagenes_alpaca = os.listdir(os.path.join(nueva_ruta_base, 'ALPACA'))
    

    images_to_test = random.sample([img for img in lista_imagenes_alpaca if img.lower().endswith('.jpg')], 4)
    
    print(f"Imágenes seleccionadas al azar para la prueba: {images_to_test}")

except FileNotFoundError:
    
    print(f"Error: No se encontró el directorio {os.path.join(nueva_ruta_base, 'ALPACA')}")
    images_to_test = ['ALPACA_0025.jpg', 'ALPACA_0080.jpg', 'ALPACA_0150.jpg', 'ALPACA_0084.jpg']


# =============================================================================
# --- SECCIÓN 2: FUNCIÓN DE PROCESAMIENTO DE IMÁGENES ---
# =============================================================================

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    # 1. Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization).
    # Este filtro mejora el contraste en pequeñas regiones de la imagen,
    # lo que es ideal para revelar detalles finos de la textura.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # 2. Aplicar el detector de bordes Sobel en la dirección X.
    # Esto resalta los gradientes verticales (líneas horizontales), que en este
    # contexto corresponden a las fibras de la carne.
    # CV_64F: Profundidad de la imagen de salida para mayor precisión.
    # dx=1, dy=0: Derivada de primer orden en x (bordes verticales).
    # ksize=5: Tamaño del kernel del filtro.
    sobel_x = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=5)

    processed_img = cv2.convertScaleAbs(sobel_x)
    
    return img, processed_img

# =============================================================================
# --- SECCIÓN 3: PROCESAMIENTO POR LOTES Y VISUALIZACIÓN ---
# =============================================================================

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
    
        if class_name == 'LLAMA':
            try:
            
                numero = img_name.split('_')[1]
                img_name = f'LLAMA_{numero}'
            except IndexError:
            
                continue
        

        image_path = os.path.join(folder_path, img_name)

       
        if not os.path.exists(image_path):
            print(f"Advertencia: No se encontró el archivo {image_path}")
           
            ax_orig = axes[i, col_offset * 2]
            ax_orig.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray')
            ax_orig.set_title(f'No encontrado: {img_name}')
            ax_orig.axis('off')

            ax_proc = axes[i, col_offset * 2 + 1]
            ax_proc.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray')
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