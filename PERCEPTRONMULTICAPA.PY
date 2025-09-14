import matplotlib.pyplot as plt
import matplotlib.patches as patches

def dibujar_perceptron():
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    
    # Configuración de capas
    capas = [
        {"x": 3, "nombre": "CAPA DE ENTRADA", "neuronas": 4, "color": "#ADD8E6"},
        {"x": 9, "nombre": "CAPA OCULTA", "neuronas": 3, "color": "#FFFFE0"},
        {"x": 15, "nombre": "CAPA DE SALIDA", "neuronas": 2, "color": "#FFB6C1"}
    ]
    
    # Dibujar cada capa
    for capa in capas:
        x = capa["x"]
        y_centro = 7.5
        
        # Título de la capa
        ax.text(x, 13, capa["nombre"], ha='center', va='center', 
                fontsize=14, weight='bold')
        
        # Dibujar neuronas
        for i in range(capa["neuronas"]):
            y = y_centro + (i - (capa["neuronas"]-1)/2) * 2
            
            # Círculo de la neurona
            circulo = patches.Circle((x, y), 0.5, fill=True, 
                                   facecolor=capa["color"], edgecolor='black')
            ax.add_patch(circulo)
            
            # Etiqueta de la neurona
            if capa["nombre"] == "CAPA DE ENTRADA":
                texto = f"Entrada {i+1}" if i < 3 else "Entrada n"
            elif capa["nombre"] == "CAPA OCULTA":
                texto = f"Oculta {i+1}"
            else:
                texto = f"Salida {i+1}" if i < 1 else "Salida m"
                
            ax.text(x, y, texto, ha='center', va='center', fontsize=10)
    
    # Dibujar conexiones
    for i in range(capas[0]["neuronas"]):  # Entrada
        y1 = 7.5 + (i - (capas[0]["neuronas"]-1)/2) * 2
        
        for j in range(capas[1]["neuronas"]):  # Oculta
            y2 = 7.5 + (j - (capas[1]["neuronas"]-1)/2) * 2
            ax.plot([3.5, 8.5], [y1, y2], 'k-', lw=1)
            
        for k in range(capas[2]["neuronas"]):  # Salida
            y3 = 7.5 + (k - (capas[2]["neuronas"]-1)/2) * 2
            
            for j in range(capas[1]["neuronas"]):  # Desde oculta
                y2 = 7.5 + (j - (capas[1]["neuronas"]-1)/2) * 2
                ax.plot([9.5, 14.5], [y2, y3], 'k-', lw=1)
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('mi_perceptron_multicapa.png', dpi=300, bbox_inches='tight')
    plt.show()

# Ejecutar
dibujar_perceptron()