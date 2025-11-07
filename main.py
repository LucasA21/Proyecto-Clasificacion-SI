"""
Proyecto de Clasificación - Detección de Phishing
===================================================

Script principal que orquesta todo el pipeline de machine learning
para la detección de sitios web que realizan phishing.

El proceso sigue las siguientes etapas:
1. Carga y exploración del dataset
2. Preprocesamiento y verificación de datos
3. Selección de características (opcional)
4. División de datos en entrenamiento y prueba
5. Escalado de características
6. Entrenamiento de modelo base
7. Optimización de hiperparámetros
8. Evaluación y comparación de modelos
9. Visualización de resultados
"""

import os
import sys

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos propios
import preprocessing
import feature_selection
import model_training
import evaluation
import visualization

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

# Semilla de aleatoriedad para garantizar reproducibilidad (configurable por entorno)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# Directorio donde se guardarán los resultados
OUTPUT_SUBDIR = os.getenv("OUTPUT_SUBDIR", "resultados")
OUTPUT_DIR = os.path.join("Data", OUTPUT_SUBDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de selección de características (configurable por entorno)
# Si USAR_SELECCION=True, se realizará búsqueda aleatoria de características
# Si USAR_SELECCION=False, se usarán todas las características
USAR_SELECCION = os.getenv("USAR_SELECCION", "True").lower() == "true"
N_ITER_SELECCION = int(os.getenv("N_ITER_SELECCION", 500))  # Número de iteraciones para búsqueda aleatoria
K_FEATURES = int(os.getenv("K_FEATURES", 5))  # Tamaño del subset de características a seleccionar

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    """
    Función principal que ejecuta todo el pipeline de machine learning.
    """
    
    # ============================================================
    # 1. CARGA Y PREPROCESAMIENTO
    # ============================================================
    
    df, class_distribution = preprocessing.cargar_y_preprocesar_dataset(
        ruta_output=os.path.join(OUTPUT_DIR, 'dataset_preprocesado.csv')
    )
    
    # Visualizar distribución de clases
    visualization.graficar_distribucion_clases(
        class_distribution,
        os.path.join(OUTPUT_DIR, 'distribucion_clases.png')
    )
    
    # ============================================================
    # 2. SEPARAR CARACTERÍSTICAS Y VARIABLE OBJETIVO
    # ============================================================
    
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    # ============================================================
    # 3. DIVISIÓN DE DATOS
    # ============================================================
    
    X_train, X_test, y_train, y_test = model_training.dividir_datos(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=True
    )
    
    # ============================================================
    # 4. SELECCIÓN DE CARACTERÍSTICAS (OPCIONAL)
    # ============================================================
    # Nota: La selección se hace solo sobre el conjunto de entrenamiento
    # para evitar data leakage. El conjunto de prueba se usa solo para
    # evaluación final del modelo.
    
    if USAR_SELECCION:
        # Realizar selección de características usando solo datos de entrenamiento
        # Para validación interna, dividimos temporalmente el conjunto de entrenamiento
        from sklearn.model_selection import train_test_split
        X_sel_train, X_sel_val, y_sel_train, y_sel_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
        
        features_seleccionadas, resultados_seleccion = feature_selection.seleccionar_caracteristicas(
            X_sel_train, y_sel_train,
            X_sel_val, y_sel_val,
            usar_seleccion=True,
            n_iter=N_ITER_SELECCION,
            k=K_FEATURES,
            random_state=RANDOM_STATE,
            ruta_output=OUTPUT_DIR
        )
        
        # Filtrar características seleccionadas en ambos conjuntos
        X_train = X_train[features_seleccionadas]
        X_test = X_test[features_seleccionadas]
        
        print(f"\n✓ Se seleccionaron {len(features_seleccionadas)} características de {df.shape[1]-1} disponibles")
        print(f"  Características seleccionadas: {features_seleccionadas}")
    else:
        print("\n✓ No se realizó selección de características")
        print(f"  Se utilizarán todas las {X.shape[1]} características disponibles")
    
    # ============================================================
    # 5. ESCALADO DE CARACTERÍSTICAS
    # ============================================================
    # El escalado se realiza después de la selección para asegurar
    # que solo se escalen las características seleccionadas.
    
    X_train_scaled, X_test_scaled, scaler = model_training.escalar_caracteristicas(
        X_train, X_test
    )
    
    # Guardar scaler
    model_training.guardar_scaler(
        scaler,
        os.path.join(OUTPUT_DIR, 'scaler.joblib')
    )
    
    # ============================================================
    # 6. MODELO BASE
    # ============================================================
    
    model_base = model_training.entrenar_modelo_base(X_train_scaled, y_train)
    
    # Realizar predicciones con modelo base
    y_pred_base = model_base.predict(X_test_scaled)
    y_pred_proba_base = model_base.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluar modelo base
    resultados_base = evaluation.generar_reporte_completo(
        y_test, y_pred_base, y_pred_proba_base,
        nombre_modelo="Modelo Base"
    )
    
    # Guardar resultados para comparación
    resultados_base_dict = {
        'y_pred': y_pred_base,
        'y_pred_proba': y_pred_proba_base
    }
    
    # ============================================================
    # 7. OPTIMIZACIÓN DE HIPERPARÁMETROS
    # ============================================================
    
    resultados_optimizacion = model_training.optimizar_hiperparametros(
        X_train_scaled, y_train,
        random_state=RANDOM_STATE,
        var_smoothing_range=None,  # Usa valores por defecto
        n_splits=5
    )
    
    best_model = resultados_optimizacion['best_model']
    
    # Guardar modelo optimizado
    model_training.guardar_modelo(
        best_model,
        os.path.join(OUTPUT_DIR, 'modelo_mejorado.joblib')
    )
    
    # Realizar predicciones con modelo optimizado
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluar modelo optimizado
    resultados_optimizado = evaluation.generar_reporte_completo(
        y_test, y_pred_best, y_pred_proba_best,
        nombre_modelo="Modelo Optimizado"
    )
    
    # Guardar resultados para comparación
    resultados_optimizado_dict = {
        'y_pred': y_pred_best,
        'y_pred_proba': y_pred_proba_best
    }
    
    # ============================================================
    # 8. COMPARACIÓN DE MODELOS
    # ============================================================
    
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 60)
    
    comparacion = evaluation.comparar_modelos(
        y_test,
        resultados_base_dict,
        resultados_optimizado_dict
    )
    
    print("\n" + comparacion.to_string())
    
    # Guardar comparación de métricas
    comparacion.to_csv(os.path.join(OUTPUT_DIR, 'comparacion_metricas.csv'))
    print(f"\n✓ Comparación de métricas guardada en: {os.path.join(OUTPUT_DIR, 'comparacion_metricas.csv')}")
    
    # ============================================================
    # 9. VISUALIZACIONES
    # ============================================================
    
    # Matrices de confusión
    cm_base = resultados_base['matriz_confusion']['matriz']
    cm_optimizado = resultados_optimizado['matriz_confusion']['matriz']
    
    visualization.graficar_matrices_confusion(
        cm_base, cm_optimizado,
        os.path.join(OUTPUT_DIR, 'matriz_confusion.png')
    )
    
    # Curvas ROC
    if resultados_base['curva_roc'] and resultados_optimizado['curva_roc']:
        visualization.graficar_curva_roc(
            resultados_base['curva_roc'],
            resultados_optimizado['curva_roc'],
            os.path.join(OUTPUT_DIR, 'curva_roc.png')
        )
    
    # ============================================================
    # 10. RESUMEN FINAL
    # ============================================================
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)
    print(f"\nTodos los resultados han sido guardados en: {OUTPUT_DIR}")
    print("\nArchivos generados:")
    print("  - dataset_preprocesado.csv")
    print("  - distribucion_clases.png")
    if USAR_SELECCION:
        print("  - log_random_search_subsets.csv")
        print("  - frecuencias_optimos.csv")
    print("  - comparacion_metricas.csv")
    print("  - matriz_confusion.png")
    print("  - curva_roc.png")
    print("  - scaler.joblib")
    print("  - modelo_mejorado.joblib")


if __name__ == "__main__":
    main()
