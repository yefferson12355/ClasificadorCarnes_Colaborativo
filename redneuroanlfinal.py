import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from graphviz import Digraph
import numpy as np

def visualize_resnet152_architecture():
    # Crear el modelo ResNet152
    model = models.resnet152(pretrained=True)
    
    # Crear gráfico con Graphviz
    dot = Digraph(comment='ResNet152 Architecture')
    dot.attr(rankdir='TB', size='20,15', dpi='300')
    dot.attr('node', shape='box', style='filled', color='lightblue', fontname='Arial')
    
    # Capas principales
    layers = [
        ('Input\n(3x224x224)', 'conv1'),
        ('Conv1\n7x7, 64', 'bn1'),
        ('BatchNorm1', 'relu1'),
        ('ReLU', 'maxpool'),
        ('MaxPool\n3x3', 'layer1'),
        ('Layer1\n3x Bottleneck', 'layer2'),
        ('Layer2\n8x Bottleneck', 'layer3'),
        ('Layer3\n36x Bottleneck', 'layer4'),
        ('Layer4\n3x Bottleneck', 'avgpool'),
        ('AvgPool', 'fc'),
        ('Fully Connected', 'Output\n(1000 classes)')
    ]
    
    # Añadir nodos
    for i, (name, node_id) in enumerate(layers):
        if 'Bottleneck' in name:
            dot.node(node_id, name, fillcolor='lightcoral')
        elif 'Conv' in name or 'Fully' in name:
            dot.node(node_id, name, fillcolor='lightgreen')
        else:
            dot.node(node_id, name)
    
    # Conectar nodos
    for i in range(len(layers)-1):
        dot.edge(layers[i][1], layers[i+1][1])
    
    # Añadir detalles de las capas bottleneck
    with dot.subgraph(name='cluster_bottleneck') as c:
        c.attr(label='Estructura Bottleneck', style='filled', color='lightyellow', fontname='Arial')
        c.node('bt_conv1', 'Conv 1x1\nReducción canales', fillcolor='lightblue')
        c.node('bt_conv2', 'Conv 3x3\nConvolución espacial', fillcolor='lightblue')
        c.node('bt_conv3', 'Conv 1x1\nExpansión canales', fillcolor='lightblue')
        c.node('bt_skip', 'Conexión residual\n(Shortcut)', fillcolor='lightgreen')
        c.node('bt_add', 'Suma', shape='circle', fillcolor='orange')
        c.node('bt_relu', 'ReLU', fillcolor='pink')
        
        c.edges([('bt_conv1', 'bt_conv2'), ('bt_conv2', 'bt_conv3'), 
                ('bt_conv3', 'bt_add'), ('bt_skip', 'bt_add'), 
                ('bt_add', 'bt_relu')])
    
    # Añadir información adicional
    dot.node('info', 'ResNet152\n• 60.2M parámetros\n• 152 capas profundas\n• Conexiones residuales', 
             shape='note', fillcolor='lightgrey', fontname='Arial')
    
    return dot

def plot_layer_parameters():
    # Distribución de parámetros por tipo de capa
    layer_types = ['Convolucional', 'BatchNorm', 'Fully Connected', 'Otros']
    parameters = [58.2, 0.8, 1.2, 0.0]  # En millones
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de pastel
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax1.pie(parameters, labels=layer_types, autopct='%1.1fM', 
                                      colors=colors, startangle=90)
    ax1.set_title('Distribución de Parámetros por Tipo de Capa')
    
    # Gráfico de barras por bloques
    blocks = ['Conv1', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'FC']
    block_params = [0.9, 2.2, 7.1, 14.9, 21.8, 1.2]
    
    bars = ax2.bar(blocks, block_params, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9c80e', '#f86624', '#ea3546'])
    ax2.set_title('Parámetros por Bloque Principal (millones)')
    ax2.set_ylabel('Millones de parámetros')
    ax2.set_xlabel('Bloques de la Red')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_feature_dimensions():
    # Evolución de las dimensiones de los features
    stages = ['Input', 'Conv1', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Output']
    channels = [3, 64, 256, 512, 1024, 2048, 1000]
    spatial_size = [224, 112, 56, 28, 14, 7, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Evolución de canales
    ax1.plot(stages, channels, 'o-', linewidth=2, markersize=8, color='#ff6b6b')
    ax1.set_title('Evolución del Número de Canales')
    ax1.set_ylabel('Número de Canales')
    ax1.set_xlabel('Etapa de la Red')
    ax1.grid(True, alpha=0.3)
    
    for i, (stage, channel) in enumerate(zip(stages, channels)):
        ax1.text(i, channel + 100, str(channel), ha='center', va='bottom', fontweight='bold')
    
    # Evolución del tamaño espacial
    ax2.plot(stages, spatial_size, 's-', linewidth=2, markersize=8, color='#4ecdc4')
    ax2.set_title('Evolución del Tamaño Espacial')
    ax2.set_ylabel('Tamaño Espacial (px)')
    ax2.set_xlabel('Etapa de la Red')
    ax2.grid(True, alpha=0.3)
    
    for i, (stage, size) in enumerate(zip(stages, spatial_size)):
        ax2.text(i, size + 10, f'{size}x{size}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Visualizar la arquitectura
print("=== DIAGRAMA DE LA ARQUITECTURA RESNET152 ===")
dot = visualize_resnet152_architecture()
dot.render('resnet152_architecture', format='png', cleanup=True)
print("Diagrama guardado como 'resnet152_architecture.png'")

# Mostrar gráficos adicionales
print("\n=== ANÁLISIS DE PARÁMETROS ===")
plot_layer_parameters()

print("\n=== EVOLUCIÓN DE DIMENSIONES ===")
plot_feature_dimensions()

# Información adicional
print("\n" + "="*50)
print("RESUMEN DE LA ARQUITECTURA RESNET152")
print("="*50)
print("• Capa inicial: Conv 7x7 con 64 filtros")
print("• 4 Bloques principales (Layer1-Layer4)")
print("• Total de capas: 152 capas profundas")
print("• Parámetros totales: 60.2 millones")
print("• Conexiones residuales: Sí")
print("• Función de activación: ReLU")
print("• Normalización: Batch Normalization")
print("• Pooling final: Average Pooling")
print("• Capa final: Fully Connected (1000 clases)")
print("="*50)