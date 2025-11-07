# Proyecto de Clasificación - Detección de Phishing

## Descripción general
Este proyecto implementa y documenta un pipeline de **Gaussian Naive Bayes** para detectar sitios web que realizan phishing. El flujo completo vive en el cuaderno `Notebooks/Clasificacion_Phishing.ipynb`, donde se ejecutan todas las etapas del proyecto: exploración del dataset, preprocesamiento, selección de características, ajuste de hiperparámetros y comparación de resultados.

El infome detallado con el paso a paso y las conclusiones se encuentra en `Docs/Informe.pdf`

**Dataset:** [Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)

## Requisitos
- Python 3.10 o superior
- pip para instalar dependencias

## Instalación rápida
```bash
cd Proyecto-Clasificacion-SI
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución del pipeline
1. Activá el entorno virtual (`source venv/bin/activate`).
2. Abrir `Notebooks/Clasificacion_Phishing.ipynb` en Jupyter Lab / Notebook.
3. Ejecutá **Run All** para reproducir los resultados.

El cuaderno genera automáticamente todos los artefactos en `Data/resultados_notebook/`:
- `distribucion_clases_notebook.png`: distribución de clases.
- `comparacion_metricas_notebook.csv`: tabla con Accuracy, Precision, Recall, F1-score y AUC-ROC (modelo base vs. optimizado).
- `frecuencias_optimos_notebook.csv` y `log_random_search_subsets_notebook.csv`: trazabilidad de la búsqueda de características.
- `roc_modelo_base.png` y `roc_comparacion_notebook.png`: curvas ROC y matrices de confusión comparativas.


## Estructura del repositorio
```
Proyecto-Clasificacion-SI/
├── Data/
│   ├── phishing+websites/           # Dataset original (ARFF)
│   └── resultados_notebook/         # Artefactos generados por el cuaderno
├── Docs/
│   ├── proyecto.md                  # Consigna del trabajo
│   └── informe.md                   # Informe breve (máx. 3 hojas)
├── Notebooks/
│   └── Clasificacion_Phishing.ipynb # Pipeline completo
├── requirements.txt                 # Dependencias
├── utils.py                         # Utilidad para cargar el ARFF
└── README.md
```

## Referencias
- UCI Machine Learning Repository – [Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- Sistemas Inteligentes 2025 – Universidad Nacional de Tierra del Fuego

