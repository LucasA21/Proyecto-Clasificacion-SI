"""
Módulo de Selección de Características
=======================================

Este módulo implementa técnicas de selección de características
usando búsqueda aleatoria (Random Search) para encontrar los
mejores subsets de variables que maximizan el desempeño del modelo.

Basado en metaheurísticas de búsqueda global basadas en Monte Carlo.
"""

import pandas as pd
import numpy as np
import os
import random
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def random_search_features(X_train, y_train, X_test, y_test, 
                          n_iter=1000, k=5, random_state=42):
    """
    Realiza búsqueda aleatoria de subsets de características para encontrar
    las mejores combinaciones que maximizan el F1-score.
    
    Esta técnica utiliza muestreo aleatorio repetido para explorar subsets
    de variables en un espacio combinatorio muy grande, aproximando una
    solución de alta calidad sin necesidad de evaluar todas las combinaciones.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Características de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    X_test : pd.DataFrame
        Características de prueba
    y_test : pd.Series
        Etiquetas de prueba
    n_iter : int, default=500
        Número de iteraciones de búsqueda aleatoria
    k : int, default=5
        Tamaño del subset de características a seleccionar
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Retorna:
    --------
    dict : Diccionario con resultados de la búsqueda
        - 'mejor_f1': Mejor F1-score encontrado
        - 'mejor_subset': Lista de características del mejor subset
        - 'frecuencias': Counter con frecuencia de características en óptimos locales
        - 'resultados': Lista de todos los resultados
    """
    random.seed(random_state)
    
    features = X_train.columns.tolist()
    mejor_f1 = 0
    mejor_subset = None
    resultados = []
    freq_optimos = Counter()
    
    print(f"\nIniciando Random Search de subsets...")
    print(f"  Iteraciones: {n_iter}")
    print(f"  Tamaño de subset: {k}")
    print(f"  Total de características: {len(features)}\n")
    
    for i in range(n_iter):
        # Seleccionar subset aleatorio de k características
        subset = random.sample(features, k)
        
        # Entrenar modelo con el subset
        model = GaussianNB()
        model.fit(X_train[subset], y_train)
        y_pred = model.predict(X_test[subset])
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        resultados.append({
            "iteracion": i + 1,
            "subset": subset,
            "f1_score": f1
        })
        
        # Actualizar mejor resultado si encontramos uno mejor
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_subset = subset
            
            # Actualizar contador de características que aparecen en óptimos locales
            freq_optimos.update(subset)
            
            print(f"Óptimo local en iteración {i + 1}/{n_iter}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  Variables: {subset}")
            print(f"  Top 5 características más frecuentes hasta ahora:")
            for feat, freq in freq_optimos.most_common(5):
                print(f"    {feat}: {freq} veces")
            print()
    
    print(f"\nBúsqueda finalizada")
    print(f"Mejor F1-score global: {mejor_f1:.4f}")
    print(f"Mejor subset encontrado: {mejor_subset}")
    
    return {
        'mejor_f1': mejor_f1,
        'mejor_subset': mejor_subset,
        'frecuencias': freq_optimos,
        'resultados': resultados
    }


def obtener_top_features(freq_optimos, k=10):
    """
    Obtiene las k características más frecuentes en los óptimos locales
    encontrados durante la búsqueda aleatoria.
    
    Parámetros:
    -----------
    freq_optimos : Counter
        Contador con frecuencia de características
    k : int, default=10
        Número de características top a retornar
    
    Retorna:
    --------
    list : Lista de las k características más frecuentes
    """
    top_features = [feature for feature, _ in freq_optimos.most_common(k)]
    return top_features


def guardar_resultados_seleccion(resultados, freq_optimos, ruta_output):
    """
    Guarda los resultados de la selección de características en archivos CSV.
    
    Parámetros:
    -----------
    resultados : list
        Lista de resultados de la búsqueda
    freq_optimos : Counter
        Contador con frecuencia de características
    ruta_output : str
        Directorio donde guardar los archivos
    """
    os.makedirs(ruta_output, exist_ok=True)
    
    # Guardar log de resultados
    df_resultados = pd.DataFrame(resultados).sort_values(
        by="f1_score", ascending=False
    )
    log_path = os.path.join(ruta_output, "log_random_search_subsets.csv")
    df_resultados.to_csv(log_path, index=False)
    
    # Guardar frecuencias de características
    freq_path = os.path.join(ruta_output, "frecuencias_optimos.csv")
    pd.DataFrame(freq_optimos.most_common()).to_csv(
        freq_path, 
        index=False,
        header=["feature", "frecuencia"]
    )
    
    print(f"\nResultados guardados en:")
    print(f"  - {log_path}")
    print(f"  - {freq_path}")


def seleccionar_caracteristicas(X_train, y_train, X_test, y_test,
                               usar_seleccion=True, n_iter=1000, k=5,
                               random_state=42, ruta_output=None):
    """
    Función principal para seleccionar características.
    Si usar_seleccion=True, realiza búsqueda aleatoria.
    Si usar_seleccion=False, retorna todas las características.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Características de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    X_test : pd.DataFrame
        Características de prueba
    y_test : pd.Series
        Etiquetas de prueba
    usar_seleccion : bool, default=True
        Si True, realiza selección de características
    n_iter : int, default=500
        Número de iteraciones para búsqueda aleatoria
    k : int, default=5
        Tamaño del subset a seleccionar
    random_state : int, default=42
        Semilla para reproducibilidad
    ruta_output : str, opcional
        Directorio donde guardar resultados
    
    Retorna:
    --------
    tuple : (caracteristicas_seleccionadas, resultados_busqueda)
        - características_seleccionadas: Lista de nombres de características
        - resultados_busqueda: Dict con resultados de la búsqueda (None si no se usa selección)
    """
    if not usar_seleccion:
        print("\nNo se realizará selección de características.")
        print("Se utilizarán todas las características disponibles.")
        return X_train.columns.tolist(), None
    
    print("\n" + "=" * 60)
    print("SELECCIÓN DE CARACTERÍSTICAS")
    print("=" * 60)
    
    # Realizar búsqueda aleatoria
    resultados_busqueda = random_search_features(
        X_train, y_train, X_test, y_test,
        n_iter=n_iter, k=k, random_state=random_state
    )
    
    # Guardar resultados si se especifica ruta
    if ruta_output:
        guardar_resultados_seleccion(
            resultados_busqueda['resultados'],
            resultados_busqueda['frecuencias'],
            ruta_output
        )
    
    # Obtener características más frecuentes
    top_features = obtener_top_features(resultados_busqueda['frecuencias'], k=k)
    
    print(f"\nTop {k} características más frecuentes en óptimos locales:")
    print(top_features)
    
    return top_features, resultados_busqueda

