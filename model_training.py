"""
Módulo de Entrenamiento de Modelos
===================================

Este módulo contiene funciones para entrenar modelos de clasificación
usando Gaussian Naive Bayes, incluyendo entrenamiento de modelo base
y optimización de hiperparámetros.
"""

import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


def dividir_datos(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    
    Parámetros:
    -----------
    X : pd.DataFrame
        Características
    y : pd.Series
        Variable objetivo
    test_size : float, default=0.2
        Proporción del dataset para prueba (20%)
    random_state : int, default=42
        Semilla para reproducibilidad
    stratify : bool, default=True
        Si True, mantiene proporción de clases en ambos conjuntos
    
    Retorna:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 60)
    print("ETAPA 3: DIVISIÓN DE DATOS")
    print("=" * 60)
    
    print(f"\nCaracterísticas (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    print("\nDistribución de clases en entrenamiento:")
    print(y_train.value_counts().sort_index())
    print("\nDistribución de clases en prueba:")
    print(y_test.value_counts().sort_index())
    
    return X_train, X_test, y_train, y_test


def escalar_caracteristicas(X_train, X_test):
    """
    Escala las características usando StandardScaler para normalizar
    a media 0 y desviación estándar 1.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Características de entrenamiento
    X_test : pd.DataFrame
        Características de prueba
    
    Retorna:
    --------
    tuple : (X_train_scaled, X_test_scaled, scaler)
        - X_train_scaled: Características escaladas de entrenamiento
        - X_test_scaled: Características escaladas de prueba
        - scaler: Objeto StandardScaler entrenado
    """
    print("\n" + "=" * 60)
    print("ETAPA 4: ESCALADO DE CARACTERÍSTICAS")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame para mantener nombres de columnas
    import pandas as pd
    X_train_scaled = pd.DataFrame(
        X_train_scaled, 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    print("\n✓ Características escaladas usando StandardScaler")
    print(f"  Media de características (debe estar cerca de 0): {X_train_scaled.mean().mean():.6f}")
    print(f"  Desviación estándar (debe estar cerca de 1): {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


def entrenar_modelo_base(X_train, y_train):
    """
    Entrena un modelo GaussianNB base sin optimización de hiperparámetros.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Características de entrenamiento escaladas
    y_train : pd.Series
        Etiquetas de entrenamiento
    
    Retorna:
    --------
    GaussianNB : Modelo entrenado
    """
    print("\n" + "=" * 60)
    print("ETAPA 6: MODELO BASE")
    print("=" * 60)
    
    print("\nEntrenando modelo base (GaussianNB con parámetros por defecto)...")
    
    model_base = GaussianNB()
    model_base.fit(X_train, y_train)
    
    print("✓ Modelo base entrenado")
    
    return model_base


def optimizar_hiperparametros(X_train, y_train, random_state=42, 
                              var_smoothing_range=None, n_splits=5):
    """
    Optimiza los hiperparámetros del modelo usando GridSearchCV.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Características de entrenamiento escaladas
    y_train : pd.Series
        Etiquetas de entrenamiento
    random_state : int, default=42
        Semilla para reproducibilidad
    var_smoothing_range : tuple, opcional
        (inicio, fin, num_valores) para el rango de var_smoothing.
        Si None, usa valores por defecto: np.logspace(-9, -3, 10)
    n_splits : int, default=5
        Número de folds en validación cruzada
    
    Retorna:
    --------
    dict : Diccionario con resultados de la optimización
        - 'best_model': Mejor modelo encontrado
        - 'best_params': Mejores parámetros
        - 'best_score': Mejor score en validación cruzada
        - 'grid_search': Objeto GridSearchCV completo
    """
    print("\n" + "=" * 60)
    print("ETAPA 7: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 60)
    
    # Definir espacio de búsqueda
    if var_smoothing_range is None:
        var_smoothing_values = np.logspace(-9, -3, 10)
    else:
        var_smoothing_values = np.logspace(
            var_smoothing_range[0],
            var_smoothing_range[1],
            var_smoothing_range[2]
        )
    
    param_grid = {
        'var_smoothing': var_smoothing_values
    }
    
    print(f"\nEspacio de búsqueda para var_smoothing:")
    print(f"  Valores: {var_smoothing_values}")
    
    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print("\nIniciando búsqueda de hiperparámetros con GridSearchCV...")
    print("  Métrica de optimización: F1-score")
    print(f"  Validación cruzada: StratifiedKFold ({n_splits} folds)")
    
    # Realizar búsqueda
    grid_search = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n✓ Optimización completada")
    print(f"\nMejor parámetro encontrado:")
    print(f"  var_smoothing = {grid_search.best_params_['var_smoothing']:.2e}")
    print(f"  Mejor F1-score en validación cruzada: {grid_search.best_score_:.4f}")
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search': grid_search
    }


def guardar_modelo(modelo, ruta):
    """
    Guarda un modelo entrenado en disco.
    
    Parámetros:
    -----------
    modelo : sklearn model
        Modelo a guardar
    ruta : str
        Ruta donde guardar el modelo
    """
    os.makedirs(os.path.dirname(ruta) if os.path.dirname(ruta) else '.', exist_ok=True)
    joblib.dump(modelo, ruta)
    print(f"✓ Modelo guardado en: {ruta}")


def guardar_scaler(scaler, ruta):
    """
    Guarda un scaler en disco.
    
    Parámetros:
    -----------
    scaler : StandardScaler
        Scaler a guardar
    ruta : str
        Ruta donde guardar el scaler
    """
    os.makedirs(os.path.dirname(ruta) if os.path.dirname(ruta) else '.', exist_ok=True)
    joblib.dump(scaler, ruta)
    print(f"✓ Scaler guardado en: {ruta}")

