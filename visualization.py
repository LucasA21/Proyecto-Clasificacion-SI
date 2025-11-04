"""
Módulo de Visualización
========================

Este módulo contiene funciones para generar visualizaciones
de los resultados del análisis y evaluación de modelos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def configurar_estilo():
    """
    Configura el estilo de las visualizaciones.
    """
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")


def graficar_distribucion_clases(class_distribution, ruta_output):
    """
    Genera una gráfica de barras mostrando la distribución de clases.
    
    Parámetros:
    -----------
    class_distribution : pd.Series
        Distribución de clases (valores y conteos)
    ruta_output : str
        Ruta donde guardar la gráfica
    """
    configurar_estilo()
    
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.title('Distribución de Clases en el Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks([0, 1], ['Legítimo (-1)', 'Phishing (1)'], rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(ruta_output) if os.path.dirname(ruta_output) else '.', exist_ok=True)
    plt.savefig(ruta_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfica de distribución guardada en: {ruta_output}")


def graficar_matrices_confusion(cm_base, cm_optimizado, ruta_output):
    """
    Genera una visualización comparativa de matrices de confusión.
    
    Parámetros:
    -----------
    cm_base : array-like
        Matriz de confusión del modelo base
    cm_optimizado : array-like
        Matriz de confusión del modelo optimizado
    ruta_output : str
        Ruta donde guardar la gráfica
    """
    configurar_estilo()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz de confusión - Modelo Base
    sns.heatmap(
        cm_base, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        ax=axes[0],
        xticklabels=['Legítimo (-1)', 'Phishing (1)'],
        yticklabels=['Legítimo (-1)', 'Phishing (1)']
    )
    axes[0].set_title('Modelo Base', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicción', fontsize=12)
    axes[0].set_ylabel('Real', fontsize=12)
    
    # Matriz de confusión - Modelo Optimizado
    sns.heatmap(
        cm_optimizado, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        ax=axes[1],
        xticklabels=['Legítimo (-1)', 'Phishing (1)'],
        yticklabels=['Legítimo (-1)', 'Phishing (1)']
    )
    axes[1].set_title('Modelo Optimizado', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicción', fontsize=12)
    axes[1].set_ylabel('Real', fontsize=12)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(ruta_output) if os.path.dirname(ruta_output) else '.', exist_ok=True)
    plt.savefig(ruta_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Matrices de confusión guardadas en: {ruta_output}")


def graficar_curva_roc(roc_base, roc_optimizado, ruta_output):
    """
    Genera una gráfica comparativa de curvas ROC.
    
    Parámetros:
    -----------
    roc_base : dict
        Diccionario con resultados ROC del modelo base
        Debe contener: 'fpr', 'tpr', 'auc'
    roc_optimizado : dict
        Diccionario con resultados ROC del modelo optimizado
        Debe contener: 'fpr', 'tpr', 'auc'
    ruta_output : str
        Ruta donde guardar la gráfica
    """
    configurar_estilo()
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(
        roc_base['fpr'], 
        roc_base['tpr'], 
        label=f'Modelo Base (AUC = {roc_base["auc"]:.4f})', 
        linewidth=2
    )
    plt.plot(
        roc_optimizado['fpr'], 
        roc_optimizado['tpr'], 
        label=f'Modelo Optimizado (AUC = {roc_optimizado["auc"]:.4f})', 
        linewidth=2
    )
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio (AUC = 0.50)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=12)
    plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(ruta_output) if os.path.dirname(ruta_output) else '.', exist_ok=True)
    plt.savefig(ruta_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Curva ROC guardada en: {ruta_output}")

