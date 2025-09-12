# fusion_final_con_imagenes.py
# Predice con CNN y XGBoost por separado, fusiona por promedio.
# MUESTRA Y GUARDA IMÁGENES CON ETIQUETAS SUPERPUESTAS.

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import xgboost as xgb
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ----------- CONFIGURAR RUTAS -----------
ruta_modelo_keras = "clasificador_carne_resnet50_con_procesamiento_integrado.keras"
ruta_modelo_xgb = "xgboost_model.json"
ruta_imagenes = "\CNN_RNN_FUS\imgvalidation"
ruta_resultados = "resultados_visuales"  # Carpeta donde se guardarán imágenes etiquetadas

# Crear carpeta de resultados si no existe
os.makedirs(ruta_resultados, exist_ok=True)

tamaño_imagen = (224, 224)

# ----------- ETIQUETAS -----------
clases = ["LLAMA", "ALPACA"]

# ----------- CARGAR MODELOS -----------
print("📦 Cargando modelos...")

modelo_cnn = load_model(ruta_modelo_keras)
print("✅ Modelo CNN cargado.")

modelo_xgb = xgb.Booster()
modelo_xgb.load_model(ruta_modelo_xgb)
print(f"✅ Modelo XGBoost cargado. Espera {modelo_xgb.num_features()} características.")

# ----------- EXTRAER ETIQUETA REAL -----------
def obtener_etiqueta_real(nombre_archivo):
    nombre = nombre_archivo.upper()
    if "LLAMA" in nombre:
        return "LLAMA"
    elif "ALPACA" in nombre:
        return "ALPACA"
    else:
        return None

# ----------- PREPROCESAR IMAGEN PARA CNN -----------
def cargar_y_preprocesar_cnn(ruta_img, tamaño=tamaño_imagen):
    try:
        img = image.load_img(ruta_img, target_size=tamaño)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array, img  # Devuelve array para predicción + objeto PIL para visualización
    except Exception as e:
        print(f"❌ Error CNN en {ruta_img}: {e}")
        return None, None

# ----------- PREPROCESAR IMAGEN PARA XGBOOST (SOLO 3 FEATURES: R, G, B PROMEDIO) -----------
def extraer_features_para_xgb(ruta_img, tamaño=tamaño_imagen):
    try:
        img = image.load_img(ruta_img, target_size=tamaño)
        img_array = image.img_to_array(img)

        avg_r = np.mean(img_array[:, :, 0]) / 255.0
        avg_g = np.mean(img_array[:, :, 1]) / 255.0
        avg_b = np.mean(img_array[:, :, 2]) / 255.0

        return np.array([[avg_r, avg_g, avg_b]])
    except Exception as e:
        print(f"❌ Error al extraer features para XGBoost en {ruta_img}: {e}")
        return None

# ----------- PREDECIR CON XGBOOST -----------
def predecir_con_xgb(features):
    try:
        dmatrix = xgb.DMatrix(features)
        pred_raw = modelo_xgb.predict(dmatrix)[0]

        if isinstance(pred_raw, (float, np.float32, np.float64)):
            return np.array([1 - pred_raw, pred_raw])
        else:
            return pred_raw
    except Exception as e:
        print(f"❌ Error al predecir con XGBoost: {e}")
        return None

# ----------- ETIQUETAR Y GUARDAR IMAGEN -----------
def etiquetar_y_guardar_imagen(img_pil, archivo, etiqueta_real, clase_cnn, conf_cnn, clase_xgb, conf_xgb, clase_fusion, conf_fusion, fusion_correcta, ruta_guardado):
    try:
        # Convertir a RGB si es necesario
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # Crear imagen más grande para añadir texto debajo
        ancho, alto = img_pil.size
        nueva_altura = alto + 180  # Espacio extra para texto
        img_con_texto = Image.new("RGB", (ancho, nueva_altura), "white")
        img_con_texto.paste(img_pil, (0, 0))

        # Preparar para dibujar texto
        draw = ImageDraw.Draw(img_con_texto)

        # Intentar cargar fuente, si falla usa default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Coordenadas iniciales para texto
        y_offset = alto + 10

        # Escribir información
        if etiqueta_real:
            draw.text((10, y_offset), f"Etiqueta Real: {etiqueta_real}", fill="black", font=font)
            y_offset += 25

        draw.text((10, y_offset), f"CNN: {clase_cnn} ({conf_cnn:.3f})", fill="blue", font=font)
        y_offset += 25

        draw.text((10, y_offset), f"XGBoost: {clase_xgb} ({conf_xgb:.3f})", fill="green", font=font)
        y_offset += 25

        color_fusion = "green" if fusion_correcta else "red" if fusion_correcta is not None else "black"
        simbolo = " ✅" if fusion_correcta else " ❌" if fusion_correcta is not None else ""
        draw.text((10, y_offset), f"FUSIÓN: {clase_fusion} ({conf_fusion:.3f}){simbolo}", fill=color_fusion, font=font)

        # Guardar
        nombre_guardado = os.path.join(ruta_guardado, f"resultado_{archivo}")
        img_con_texto.save(nombre_guardado)
        print(f"   🖼️  Imagen guardada: {nombre_guardado}")

    except Exception as e:
        print(f"❌ Error al etiquetar/guardar imagen {archivo}: {e}")

# ----------- PROCESAR IMÁGENES -----------
print(f"\n🔍 Procesando imágenes en '{ruta_imagenes}'...")
resultados = []

for archivo in os.listdir(ruta_imagenes):
    if not archivo.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    etiqueta_real = obtener_etiqueta_real(archivo)
    ruta_img = os.path.join(ruta_imagenes, archivo)

    # --- Predicción CNN ---
    img_array, img_pil = cargar_y_preprocesar_cnn(ruta_img)
    if img_array is None or img_pil is None:
        continue
    pred_cnn = modelo_cnn.predict(img_array, verbose=0)[0]

    # --- Predicción XGBoost ---
    features_xgb = extraer_features_para_xgb(ruta_img)
    if features_xgb is None:
        continue
    pred_xgb = predecir_con_xgb(features_xgb)
    if pred_xgb is None:
        continue

    # --- FUSIÓN POR PROMEDIO ---
    pred_fusion = (pred_cnn + pred_xgb) / 2.0

    # --- Obtener clases y confianzas ---
    clase_cnn = clases[np.argmax(pred_cnn)]
    clase_xgb = clases[np.argmax(pred_xgb)]
    clase_fusion = clases[np.argmax(pred_fusion)]

    conf_cnn = np.max(pred_cnn)
    conf_xgb = np.max(pred_xgb)
    conf_fusion = np.max(pred_fusion)

    fusion_correcta = None
    if etiqueta_real:
        fusion_correcta = (clase_fusion == etiqueta_real)

    # --- ETIQUETAR Y GUARDAR IMAGEN ---
    etiquetar_y_guardar_imagen(
        img_pil, archivo, etiqueta_real,
        clase_cnn, conf_cnn,
        clase_xgb, conf_xgb,
        clase_fusion, conf_fusion,
        fusion_correcta, ruta_resultados
    )

    # --- Guardar en resultados para CSV ---
    resultados.append({
        "Imagen": archivo,
        "Etiqueta_Real": etiqueta_real or "Desconocida",
        "CNN_Pred": clase_cnn,
        "CNN_Conf": conf_cnn,
        "XGB_Pred": clase_xgb,
        "XGB_Conf": conf_xgb,
        "Fusion_Pred": clase_fusion,
        "Fusion_Conf": conf_fusion,
        "Fusion_Correcta": fusion_correcta if etiqueta_real else "N/A"
    })

    # --- Mostrar en consola ---
    print(f"\n📄 {archivo}")
    if etiqueta_real:
        print(f"   Real: {etiqueta_real}")
    print(f"   CNN:        {clase_cnn} ({conf_cnn:.3f})")
    print(f"   XGBoost:    {clase_xgb} ({conf_xgb:.3f})")
    print(f"   FUSIÓN 🧠:  {clase_fusion} ({conf_fusion:.3f}) {'✅' if fusion_correcta else '' if fusion_correcta is None else '❌'}")

# ----------- RESUMEN -----------
fusion_validas = [r for r in resultados if r["Fusion_Correcta"] != "N/A"]
if fusion_validas:
    aciertos = sum(1 for r in fusion_validas if r["Fusion_Correcta"])
    total = len(fusion_validas)
    print(f"\n📊 Exactitud de fusión: {aciertos/total:.2%} ({aciertos}/{total})")

    # Mostrar resumen visual
    plt.figure(figsize=(8, 5))
    models = ['CNN', 'XGBoost', 'Fusión']
    aciertos_cnn = sum(1 for r in fusion_validas if r["CNN_Pred"] == r["Etiqueta_Real"])
    aciertos_xgb = sum(1 for r in fusion_validas if r["XGB_Pred"] == r["Etiqueta_Real"])
    aciertos_fusion = aciertos

    plt.bar(models, [aciertos_cnn/total, aciertos_xgb/total, aciertos_fusion/total], color=['blue', 'green', 'purple'])
    plt.title("Comparación de Exactitud")
    plt.ylabel("Exactitud")
    plt.ylim(0, 1)
    for i, v in enumerate([aciertos_cnn/total, aciertos_xgb/total, aciertos_fusion/total]):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')

    plt.savefig(os.path.join(ruta_resultados, "comparacion_exactitud.png"))
    plt.show()

# ----------- GUARDAR RESULTADOS -----------
df = pd.DataFrame(resultados)
df.to_csv(os.path.join(ruta_resultados, "fusion_final_resultados.csv"), index=False, float_format="%.4f")
print(f"\n📁 Resultados guardados en '{ruta_resultados}/'")

print("\n🎉 ¡Fusión completada con éxito! Imágenes etiquetadas guardadas en la carpeta 'resultados_visuales'.")