import pandas as pd
import matplotlib.pyplot as plt

# --- Datos de la Fase 1: Entrenamiento de la Cabeza ---
phase1_data = {
    'Epoch': list(range(1, 26)),
    'Train Loss': [0.7031, 0.6535, 0.6516, 0.6417, 0.6419, 0.6151, 0.6253, 0.6016, 0.5874, 0.6030, 0.5988, 0.5836, 0.6112, 0.6009, 0.5853, 0.5841, 0.5805, 0.5791, 0.5643, 0.5628, 0.5624, 0.5539, 0.5679, 0.5684, 0.5733],
    'Train Acc': [53.80, 61.25, 60.80, 63.49, 61.85, 67.36, 65.57, 67.81, 70.34, 66.17, 69.45, 68.11, 67.21, 66.77, 70.79, 68.70, 69.15, 67.81, 70.49, 70.79, 69.30, 70.64, 72.13, 70.19, 72.28],
    'Val Loss': [0.6883, 0.5950, 0.5905, 0.5918, 0.5536, 0.5924, 0.5884, 0.5369, 0.5547, 0.5392, 0.5855, 0.5826, 0.5232, 0.5301, 0.5410, 0.5395, 0.5808, 0.5334, 0.6009, 0.5457, 0.5249, 0.5799, 0.5548, 0.5308, 0.5509],
    'Val Acc': [50.60, 69.05, 72.02, 70.83, 72.02, 66.07, 65.48, 72.62, 69.05, 73.81, 71.43, 73.21, 74.40, 72.02, 76.79, 71.43, 68.45, 77.38, 66.67, 73.81, 76.79, 70.83, 75.00, 77.38, 73.21]
}
df_phase1 = pd.DataFrame(phase1_data)

# --- Datos de la Fase 2: Fine-Tuning ---
phase2_data = {
    'Epoch': list(range(1, 26)),
    'Train Loss': [0.5472, 0.5326, 0.5173, 0.4701, 0.4915, 0.4582, 0.4270, 0.4206, 0.4205, 0.3991, 0.3519, 0.3344, 0.3660, 0.3058, 0.3162, 0.3133, 0.2688, 0.2623, 0.2532, 0.2670, 0.2750, 0.2595, 0.2406, 0.2128, 0.2039],
    'Train Acc': [73.17, 73.17, 73.62, 77.35, 75.86, 77.65, 80.63, 81.52, 79.88, 80.48, 84.20, 85.84, 84.65, 87.78, 86.89, 84.05, 89.87, 89.72, 89.42, 89.27, 88.08, 88.82, 88.82, 92.10, 91.06],
    'Val Loss': [0.5571, 0.4938, 0.4870, 0.4500, 0.4879, 0.5040, 0.4486, 0.4490, 0.4527, 0.4737, 0.4697, 0.4581, 0.4429, 0.4816, 0.4106, 0.4093, 0.4855, 0.5080, 0.3783, 0.4413, 0.3791, 0.3228, 0.4249, 0.3660, 0.3681],
    'Val Acc': [72.62, 79.17, 76.79, 80.95, 76.79, 77.98, 82.14, 79.17, 77.38, 77.98, 77.38, 81.55, 82.14, 79.17, 85.12, 85.71, 79.76, 78.57, 86.31, 82.74, 83.93, 88.10, 82.74, 83.93, 87.50]
}
df_phase2 = pd.DataFrame(phase2_data)

# --- Creación de Gráficas ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis de Métricas de Entrenamiento', fontsize=16)

# Gráfica 1: Pérdida - Fase 1
ax1.plot(df_phase1['Epoch'], df_phase1['Train Loss'], label='Pérdida de Entrenamiento', marker='o')
ax1.plot(df_phase1['Epoch'], df_phase1['Val Loss'], label='Pérdida de Validación', marker='x')
ax1.set_title('Fase 1: Entrenamiento de Cabeza - Pérdida')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Pérdida (Loss)')
ax1.legend()
ax1.grid(True)

# Gráfica 2: Precisión - Fase 1
ax2.plot(df_phase1['Epoch'], df_phase1['Train Acc'], label='Precisión de Entrenamiento', marker='o')
ax2.plot(df_phase1['Epoch'], df_phase1['Val Acc'], label='Precisión de Validación', marker='x')
ax2.set_title('Fase 1: Entrenamiento de Cabeza - Precisión')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Precisión (%)')
ax2.legend()
ax2.grid(True)

# Gráfica 3: Pérdida - Fase 2
ax3.plot(df_phase2['Epoch'], df_phase2['Train Loss'], label='Pérdida de Entrenamiento', marker='o')
ax3.plot(df_phase2['Epoch'], df_phase2['Val Loss'], label='Pérdida de Validación', marker='x')
ax3.set_title('Fase 2: Fine-Tuning - Pérdida')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Pérdida (Loss)')
ax3.legend()
ax3.grid(True)

# Gráfica 4: Precisión - Fase 2
ax4.plot(df_phase2['Epoch'], df_phase2['Train Acc'], label='Precisión de Entrenamiento', marker='o')
ax4.plot(df_phase2['Epoch'], df_phase2['Val Acc'], label='Precisión de Validación', marker='x')
ax4.set_title('Fase 2: Fine-Tuning - Precisión')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Precisión (%)')
ax4.legend()
ax4.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Cálculo y Muestra de Estadísticas Descriptivas ---
print("--- Estadísticas Fase 1 ---")
print(df_phase1[['Val Loss', 'Val Acc']].describe())
print("\n--- Estadísticas Fase 2 ---")
print(df_phase2[['Val Loss', 'Val Acc']].describe())