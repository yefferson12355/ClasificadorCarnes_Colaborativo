import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random

def simulate_training(model_name, base_accuracy, base_loss, stability):
    """
    Función para simular un proceso de entrenamiento y devolver
    las métricas finales.
    """
    print(f"\n==================================================")
    print(f"Simulando entrenamiento para el modelo: {model_name}")
    print("==================================================")
    
    # Simula el tiempo de entrenamiento
    training_time = random.uniform(5, 15)
    print(f"Epoch 1/10...")
    time.sleep(training_time / 5)
    print(f"Epoch 5/10...")
    time.sleep(training_time / 5)
    print(f"Epoch 10/10...")
    time.sleep(training_time / 5)
    
    # Genera resultados finales basados en tus datos reales
    final_accuracy = base_accuracy + random.uniform(-stability, stability)
    final_loss = base_loss + random.uniform(-stability, stability)
    
    print(f"Simulación completada en {training_time:.1f} segundos.")
    print(f"Resultado final -> Precisión: {final_accuracy:.2f}%, Pérdida: {final_loss:.4f}")
    
    return final_accuracy, final_loss

# --- 1. SIMULACIÓN DE ENTRENAMIENTO PARA CADA MODELO ---
# Usamos los datos de tus experimentos como base para la simulación.

model_names = ['Custom CNN', 'VGG16', 'ResNet50', 'ResNet152', 'InceptionV3', 'EfficientNet']
# AQUÍ ESTÁ EL CAMBIO: ResNet152 ahora tiene una precisión base de 85.0
base_accuracies = [76.3, 73.4, 72.5, 85.0, 63.9, 54.4] 
base_losses = [0.4886, 0.5584, 0.5218, 0.5671, 0.6525, 0.6792]
stabilities = [1.5, 2.0, 1.0, 2.5, 3.0, 4.0] # Simula la estabilidad de cada modelo

final_results = {'Modelo': [], 'Precisión (%)': [], 'Pérdida': []}

for i, name in enumerate(model_names):
    acc, loss = simulate_training(name, base_accuracies[i], base_losses[i], stabilities[i])
    final_results['Modelo'].append(name)
    final_results['Precisión (%)'].append(acc)
    final_results['Pérdida'].append(loss)

print("\n\n✅ Simulación de todos los entrenamientos completada.")

# --- 2. GENERACIÓN DEL GRÁFICO COMPARATIVO ---
# Este código es el mismo que usamos para visualizar tus resultados reales.

df = pd.DataFrame(final_results)

# Crear la figura y los subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Comparación de Rendimiento de Modelos (Simulado)', fontsize=18, weight='bold')

# --- Gráfico de Precisión (Ordenado de Mayor a Menor) ---
df_sorted_acc = df.sort_values('Precisión (%)', ascending=False)
bars_acc = ax[0].bar(df_sorted_acc['Modelo'], df_sorted_acc['Precisión (%)'], color='skyblue')
ax[0].set_title('Precisión de Validación Final', fontsize=14)
ax[0].set_ylabel('Precisión (%)', fontsize=12)
ax[0].set_ylim(min(df['Precisión (%)']) - 5, max(df['Precisión (%)']) + 5)
ax[0].tick_params(axis='x', rotation=45)

# Añadir etiquetas de datos
for bar in bars_acc:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

# --- Gráfico de Pérdida (Ordenado de Menor a Mayor) ---
df_sorted_loss = df.sort_values('Pérdida', ascending=True)
bars_loss = ax[1].bar(df_sorted_loss['Modelo'], df_sorted_loss['Pérdida'], color='salmon')
ax[1].set_title('Pérdida de Validación Final', fontsize=14)
ax[1].set_ylabel('Pérdida', fontsize=12)
ax[1].tick_params(axis='x', rotation=45)

# Añadir etiquetas de datos
for bar in bars_loss:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()