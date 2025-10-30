"""
Utilidades para cargar el dataset de Phishing desde archivos locales
"""
import pandas as pd
import numpy as np
import os
from scipy.io import arff


def cargar_dataset_phishing():
    """
    Carga el dataset de Phishing completo desde el archivo ARFF local
    en Data/phishing+websites/Training Dataset.arff
    
    Returns:
        pd.DataFrame: Dataset de Phishing completo con todas las columnas
    """
    # Obtener la ruta del directorio del proyecto (donde está utils.py)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir la ruta al archivo de datos
    data_path = os.path.join(project_dir, 'Data', 'phishing+websites', 'Training Dataset.arff')
    
    # Cargar el archivo ARFF
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    
    # Conversión de atributos categóricos leídos como bytes
    df = df.apply(lambda col: col.str.decode('utf-8') if col.dtype == 'object' else col)
    
    # Convertir la columna Result a entero
    df["Result"] = df["Result"].astype(int)
    
    return df


def obtener_metadatos_dataset():
    """
    Obtiene los metadatos del dataset de Phishing
    
    Returns:
        dict: Diccionario con los metadatos del dataset
    """
    # Obtener la ruta del directorio del proyecto
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir la ruta al archivo de datos
    data_path = os.path.join(project_dir, 'Data', 'phishing+websites', 'Training Dataset.arff')
    
    # Cargar solo los metadatos
    _, meta = arff.loadarff(data_path)
    
    return meta

