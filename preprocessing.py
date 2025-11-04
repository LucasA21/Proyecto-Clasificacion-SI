"""
Módulo de Preprocesamiento de Datos
====================================

Este módulo contiene funciones para cargar, explorar y preprocesar
el dataset de phishing antes de la modelación.
"""

import pandas as pd
import numpy as np
import os
from utils import cargar_dataset_phishing


def explorar_dataset(df):
    """
    Realiza una exploración inicial del dataset mostrando información
    sobre su estructura, tipos de datos y dimensiones.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a explorar
    
    Retorna:
    --------
    dict : Diccionario con información del dataset
    """
    info = {
        'shape': df.shape,
        'columnas': df.columns.tolist(),
        'tipos_datos': df.dtypes.value_counts().to_dict(),
        'valores_unicos_result': sorted(df['Result'].unique()) if 'Result' in df.columns else None
    }
    
    print(f"Dimensiones del dataset: {df.shape}")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("\nTipos de datos por columna:")
    print(df.dtypes.value_counts())
    print("\nPrimeras 5 filas del dataset:")
    print(df.head())
    
    return info


def verificar_codificacion_binaria(df, columna='Result'):
    """
    Verifica que la variable objetivo tenga codificación binaria
    coherente con el problema (1 para phishing, -1 para legítimos).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a verificar
    columna : str, default='Result'
        Nombre de la columna objetivo
    
    Retorna:
    --------
    bool : True si la codificación es correcta, False en caso contrario
    """
    valores_unicos = set(df[columna].unique())
    esperado = {-1, 1}
    
    if valores_unicos == esperado:
        print(f"\n✓ Codificación binaria verificada: -1 (legítimo), 1 (phishing)")
        return True
    else:
        print(f"\n⚠ Advertencia: La codificación binaria no es la esperada")
        print(f"Valores encontrados: {sorted(valores_unicos)}")
        return False


def verificar_valores_faltantes(df):
    """
    Verifica y reporta valores faltantes en el dataset.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a verificar
    
    Retorna:
    --------
    pd.Series : Series con conteo de valores faltantes por columna
    """
    missing_values = df.isnull().sum()
    missing_count = missing_values.sum()
    
    print("\nValores faltantes por columna:")
    if missing_count == 0:
        print("✓ No se encontraron valores faltantes en el dataset")
    else:
        print(missing_values[missing_values > 0])
        print(f"\nTotal de valores faltantes: {missing_count}")
    
    return missing_values


def analizar_distribucion_clases(df, columna='Result'):
    """
    Analiza la distribución de clases en el dataset.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    columna : str, default='Result'
        Nombre de la columna objetivo
    
    Retorna:
    --------
    pd.Series : Distribución de clases
    """
    class_distribution = df[columna].value_counts().sort_index()
    
    print("\nDistribución de clases:")
    print(class_distribution)
    print(f"\nProporción de clases:")
    
    total = len(df)
    for clase, count in class_distribution.items():
        porcentaje = count / total * 100
        etiqueta = "Legítimos (-1)" if clase == -1 else "Phishing (1)"
        print(f"  {etiqueta}: {porcentaje:.2f}%")
    
    return class_distribution


def cargar_y_preprocesar_dataset(ruta_output=None):
    """
    Carga el dataset y realiza todas las verificaciones de preprocesamiento.
    
    Parámetros:
    -----------
    ruta_output : str, opcional
        Ruta donde guardar el dataset preprocesado
    
    Retorna:
    --------
    pd.DataFrame : Dataset preprocesado y verificado
    """
    print("=" * 60)
    print("ETAPA 1: CARGA Y EXPLORACIÓN DEL DATASET")
    print("=" * 60)
    
    # Cargar dataset
    df = cargar_dataset_phishing()
    
    # Explorar dataset
    info = explorar_dataset(df)
    
    print("\n" + "=" * 60)
    print("ETAPA 2: PREPROCESAMIENTO Y VERIFICACIÓN")
    print("=" * 60)
    
    # Verificar codificación binaria
    verificar_codificacion_binaria(df)
    
    # Verificar valores faltantes
    verificar_valores_faltantes(df)
    
    # Analizar distribución de clases
    class_distribution = analizar_distribucion_clases(df)
    
    # Guardar dataset preprocesado si se especifica ruta
    if ruta_output:
        os.makedirs(os.path.dirname(ruta_output) if os.path.dirname(ruta_output) else '.', exist_ok=True)
        df.to_csv(ruta_output, index=False)
        print(f"\n✓ Dataset preprocesado guardado en: {ruta_output}")
    
    return df, class_distribution

