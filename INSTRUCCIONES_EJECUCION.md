# Instrucciones de Ejecución

## Cómo Ejecutar el Código

### 1. Activar el Entorno Virtual

Primero, asegúrate de activar el entorno virtual donde están instaladas las dependencias:

```bash
cd /home/lucas/Documentos/Proyectos/Facultad/Sistemas-Inteligentes/Proyecto-Clasificacion-SI
source venv/bin/activate
```

### 2. Ejecutar el Script Principal

```bash
python3 main.py
```

O si `python` está configurado para Python 3:

```bash
python main.py
```

### 3. Configuración Opcional

Antes de ejecutar, puedes modificar la configuración en `main.py`:

```python
# Línea 47: Habilitar/deshabilitar selección de características
USAR_SELECCION = True   # True para usar selección, False para usar todas las características

# Línea 48: Número de iteraciones para búsqueda aleatoria (solo si USAR_SELECCION=True)
N_ITER_SELECCION = 500  # Reducir para ejecución más rápida (ej: 100)

# Línea 49: Tamaño del subset de características a seleccionar
K_FEATURES = 5
```

**Nota**: Si `USAR_SELECCION = True`, el proceso puede tardar varios minutos debido a la búsqueda aleatoria de características. Para una ejecución más rápida, puedes:
- Cambiar `USAR_SELECCION = False` para usar todas las características
- Reducir `N_ITER_SELECCION` a 100 o menos

## Qué Esperar Durante la Ejecución

El script ejecuta las siguientes etapas en orden:

1. **ETAPA 1**: Carga y exploración del dataset
   - Muestra dimensiones y tipos de datos
   - Muestra primeras filas

2. **ETAPA 2**: Preprocesamiento y verificación
   - Verifica codificación binaria
   - Verifica valores faltantes
   - Analiza distribución de clases

3. **ETAPA 3**: División de datos
   - Divide en entrenamiento (80%) y prueba (20%)
   - Muestra distribución de clases en cada conjunto

4. **ETAPA 4**: Selección de características (si está habilitada)
   - Realiza búsqueda aleatoria
   - Muestra óptimos locales encontrados
   - Puede tardar varios minutos

5. **ETAPA 5**: Escalado de características
   - Normaliza características con StandardScaler

6. **ETAPA 6**: Modelo base
   - Entrena modelo GaussianNB sin optimización

7. **ETAPA 7**: Optimización de hiperparámetros
   - Busca mejor valor de var_smoothing
   - Puede tardar algunos minutos

8. **ETAPA 8**: Evaluación de modelos
   - Calcula métricas para ambos modelos
   - Genera reportes completos

9. **ETAPA 9**: Comparación de modelos
   - Compara modelo base vs optimizado
   - Guarda tabla de comparación

10. **ETAPA 10**: Visualizaciones
    - Genera gráficas de distribución, matrices de confusión y curvas ROC

## Cómo Verificar que Funciona Correctamente

### 1. Verificación de Salida en Consola

El script imprime información en cada etapa. Debes ver:

- ✓ Mensajes de éxito (✓) para cada operación completada
- Métricas numéricas (Accuracy, Precision, Recall, F1-score, AUC-ROC)
- Matrices de confusión con valores numéricos
- Mensaje final: "PROCESO COMPLETADO"

### 2. Verificación de Archivos Generados

Después de ejecutar, verifica que se hayan creado los siguientes archivos en `Data/resultados/`:

```
Data/resultados/
├── dataset_preprocesado.csv          # Dataset preprocesado
├── distribucion_clases.png          # Gráfica de distribución
├── comparacion_metricas.csv          # Tabla de comparación de métricas
├── matriz_confusion.png              # Matrices de confusión comparativas
├── curva_roc.png                     # Curvas ROC comparativas
├── scaler.joblib                      # Scaler guardado
└── modelo_mejorado.joblib            # Modelo optimizado guardado
```

Si `USAR_SELECCION = True`, también deberías ver:
```
├── log_random_search_subsets.csv     # Log de búsqueda aleatoria
└── frecuencias_optimos.csv            # Frecuencias de características
```

### 3. Verificación de Métricas

Abre el archivo `Data/resultados/comparacion_metricas.csv` con Excel, LibreOffice o un editor de texto. Deberías ver:

- **Accuracy**: Entre 0.85 y 0.95 (85% - 95%)
- **Precision**: Entre 0.60 y 0.75
- **Recall**: Entre 0.60 y 0.75
- **F1-score**: Entre 0.60 y 0.75
- **AUC-ROC**: Entre 0.90 y 0.98

**El modelo optimizado debería tener métricas mejores o iguales que el modelo base.**

### 4. Verificación de Visualizaciones

Abre las imágenes generadas:
- `distribucion_clases.png`: Debe mostrar dos barras (legítimos y phishing)
- `matriz_confusion.png`: Debe mostrar dos matrices de confusión lado a lado
- `curva_roc.png`: Debe mostrar dos curvas (modelo base y optimizado)

### 5. Verificación de Errores

Si el script se detiene con un error:

1. **Error de importación**: Asegúrate de que el entorno virtual está activado
2. **Error de archivo no encontrado**: Verifica que `Data/phishing+websites/Training Dataset.arff` existe
3. **Error de memoria**: Reduce `N_ITER_SELECCION` o usa `USAR_SELECCION = False`

## Tiempo Estimado de Ejecución

- **Sin selección de características** (`USAR_SELECCION = False`): ~2-5 minutos
- **Con selección de características** (`USAR_SELECCION = True, N_ITER_SELECCION = 500`): ~10-20 minutos
- **Con selección rápida** (`USAR_SELECCION = True, N_ITER_SELECCION = 100`): ~3-5 minutos

## Solución de Problemas Comunes

### Error: "ModuleNotFoundError: No module named 'pandas'"
**Solución**: Activa el entorno virtual: `source venv/bin/activate`

### Error: "FileNotFoundError: Training Dataset.arff"
**Solución**: Verifica que el archivo existe en `Data/phishing+websites/Training Dataset.arff`

### Error: "TypeError: got an unexpected keyword argument 'pos_label'"
**Solución**: Este error ya está corregido en el código. Si aparece, actualiza el archivo `evaluation.py`

### El proceso se detiene sin errores
**Solución**: Revisa la última línea impresa. Si está en la etapa de selección de características, es normal que tarde. Puedes reducir `N_ITER_SELECCION`.

## Resultados Esperados

Después de una ejecución exitosa, deberías obtener:

1. **Modelo base**: F1-score entre 0.40 y 0.50
2. **Modelo optimizado**: F1-score entre 0.60 y 0.70 (mejora significativa)
3. **AUC-ROC optimizado**: Entre 0.90 y 0.98 (excelente discriminación)
4. **Mejora en todas las métricas**: El modelo optimizado debería superar al base en todas las métricas

Estos resultados indican que:
- ✓ El modelo está aprendiendo patrones
- ✓ La optimización está funcionando
- ✓ El pipeline completo está operativo

