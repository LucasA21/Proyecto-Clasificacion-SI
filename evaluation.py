"""
Módulo de Evaluación de Modelos
=================================

Este módulo contiene funciones para evaluar el desempeño de modelos
de clasificación, calculando métricas y generando reportes.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)


def calcular_metricas(y_true, y_pred, y_pred_proba=None):
    """
    Calcula métricas de evaluación para clasificación binaria.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    y_pred_proba : array-like, opcional
        Probabilidades de predicción para clase positiva
    
    Retorna:
    --------
    dict : Diccionario con métricas calculadas
    """
    metricas = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', pos_label=1),
        'Recall': recall_score(y_true, y_pred, average='binary', pos_label=1),
        'F1-score': f1_score(y_true, y_pred, average='binary', pos_label=1)
    }
    
    # Calcular AUC si se proporcionan probabilidades
    # roc_auc_score requiere que las etiquetas sean 0 y 1, no -1 y 1
    if y_pred_proba is not None:
        # Mapear etiquetas de -1/1 a 0/1 para compatibilidad con roc_auc_score
        y_true_mapped = (y_true == 1).astype(int)
        metricas['AUC-ROC'] = roc_auc_score(y_true_mapped, y_pred_proba)
    
    return metricas


def calcular_matriz_confusion(y_true, y_pred):
    """
    Calcula la matriz de confusión y proporciona interpretación.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    
    Retorna:
    --------
    dict : Diccionario con matriz de confusión y métricas derivadas
        - 'matriz': Matriz de confusión (array 2x2)
        - 'TN': Verdaderos Negativos
        - 'FP': Falsos Positivos
        - 'FN': Falsos Negativos
        - 'TP': Verdaderos Positivos
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Extraer valores de la matriz de confusión
    # Asumiendo que las clases están ordenadas como [-1, 1]
    if len(cm) == 2:
        TN, FP = cm[0]
        FN, TP = cm[1]
    else:
        TN = FP = FN = TP = 0
    
    resultado = {
        'matriz': cm,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP
    }
    
    return resultado


def calcular_curva_roc(y_true, y_pred_proba, pos_label=1):
    """
    Calcula la curva ROC y el área bajo la curva.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred_proba : array-like
        Probabilidades de predicción para clase positiva
    pos_label : int, default=1
        Etiqueta de la clase positiva
    
    Retorna:
    --------
    dict : Diccionario con resultados de la curva ROC
        - 'fpr': Tasa de falsos positivos
        - 'tpr': Tasa de verdaderos positivos (Recall)
        - 'auc': Área bajo la curva ROC
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=pos_label)
    auc_score = auc(fpr, tpr)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_score
    }


def comparar_modelos(y_test, resultados_base, resultados_optimizado):
    """
    Compara las métricas de dos modelos y calcula las mejoras.
    
    Parámetros:
    -----------
    y_test : array-like
        Etiquetas de prueba
    resultados_base : dict
        Diccionario con predicciones y probabilidades del modelo base
        Debe contener: 'y_pred', 'y_pred_proba'
    resultados_optimizado : dict
        Diccionario con predicciones y probabilidades del modelo optimizado
        Debe contener: 'y_pred', 'y_pred_proba'
    
    Retorna:
    --------
    pd.DataFrame : DataFrame con comparación de métricas
    """
    # Calcular métricas para ambos modelos
    metricas_base = calcular_metricas(
        y_test, 
        resultados_base['y_pred'],
        resultados_base.get('y_pred_proba')
    )
    
    metricas_optimizado = calcular_metricas(
        y_test,
        resultados_optimizado['y_pred'],
        resultados_optimizado.get('y_pred_proba')
    )
    
    # Crear DataFrame comparativo
    comparacion = pd.DataFrame({
        'Modelo Base': metricas_base,
        'Modelo Optimizado': metricas_optimizado
    })
    
    # Calcular mejoras
    comparacion['Mejora'] = comparacion['Modelo Optimizado'] - comparacion['Modelo Base']
    comparacion['Mejora (%)'] = (
        comparacion['Mejora'] / comparacion['Modelo Base'] * 100
    ).round(2)
    
    return comparacion


def generar_reporte_completo(y_test, y_pred, y_pred_proba=None, 
                            nombre_modelo="Modelo"):
    """
    Genera un reporte completo de evaluación del modelo.
    
    Parámetros:
    -----------
    y_test : array-like
        Etiquetas de prueba
    y_pred : array-like
        Predicciones del modelo
    y_pred_proba : array-like, opcional
        Probabilidades de predicción
    nombre_modelo : str, default="Modelo"
        Nombre del modelo para el reporte
    
    Retorna:
    --------
    dict : Diccionario con todos los resultados de evaluación
    """
    print("\n" + "=" * 60)
    print(f"EVALUACIÓN: {nombre_modelo}")
    print("=" * 60)
    
    # Calcular métricas
    metricas = calcular_metricas(y_test, y_pred, y_pred_proba)
    print("\nMétricas:")
    for metrica, valor in metricas.items():
        print(f"  {metrica}: {valor:.4f}")
    
    # Calcular matriz de confusión
    cm_result = calcular_matriz_confusion(y_test, y_pred)
    print("\nMatriz de Confusión:")
    print(cm_result['matriz'])
    print(f"\n  Verdaderos Negativos (TN): {cm_result['TN']}")
    print(f"  Falsos Positivos (FP): {cm_result['FP']}")
    print(f"  Falsos Negativos (FN): {cm_result['FN']}")
    print(f"  Verdaderos Positivos (TP): {cm_result['TP']}")
    
    # Calcular curva ROC si hay probabilidades
    roc_result = None
    if y_pred_proba is not None:
        roc_result = calcular_curva_roc(y_test, y_pred_proba)
        print(f"\nAUC-ROC: {roc_result['auc']:.4f}")
    
    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=['Legítimo (-1)', 'Phishing (1)']
    ))
    
    return {
        'metricas': metricas,
        'matriz_confusion': cm_result,
        'curva_roc': roc_result
    }

