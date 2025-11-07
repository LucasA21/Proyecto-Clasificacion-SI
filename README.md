# Proyecto de Clasificación - Detección de Phishing

## Descripción del Proyecto

Este proyecto implementa y evalúa un sistema de clasificación para la detección de sitios web que realizan phishing utilizando el algoritmo Gaussian Naive Bayes.

**Dataset:** [Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)

## Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación

### 1. Navegar al directorio del proyecto

```bash
cd Proyecto-Clasificacion-SI
```

### 2. Crear un entorno virtual

```bash
# En Linux/Mac
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Ejecución del Pipeline

El script principal `main.py` permite configurar la ejecución a través de variables de entorno:

- `USAR_SELECCION`: define si se realiza la selección de características (`True` | `False`).
- `K_FEATURES`: cantidad de variables a seleccionar cuando `USAR_SELECCION=True`.
- `RANDOM_STATE`: semilla utilizada para la partición train/test y validaciones.
- `OUTPUT_SUBDIR`: subcarpeta dentro de `Data/` donde se guardarán los resultados.

Ejemplos de uso

```bash
# Ejecutar con todas las variables (reproduce la configuración de la profesora)
USAR_SELECCION=False RANDOM_STATE=13 OUTPUT_SUBDIR=resultados_todas_vars python3 main.py

# Ejecutar con selección de variables (pipeline propuesto)
USAR_SELECCION=True K_FEATURES=5 RANDOM_STATE=42 OUTPUT_SUBDIR=resultados_nuestros python3 main.py
```

Todos los archivos generados (métricas, visualizaciones, modelos) se almacenan en `Data/<OUTPUT_SUBDIR>/`.

## Estructura del Proyecto

```
Proyecto-Clasificacion-SI/
├── Data/
│   └── phishing+websites/
│       ├── Training Dataset.arff    # Dataset principal
│       └── Phishing Websites Features.docx
├── Algoritmos/
│   ├── Phishing.py                 # Script principal con optimización
│   └── MejoresFeatures.py          # Búsqueda de mejores features
├── Docs/
│   └── proyecto.md                 # Especificaciones del proyecto
├── Data/
│   └── resultados/                # Directorio con resultados (generado)
├── utils.py                        # Utilidades para cargar datos
├── main.py                         # Script principal recomendado
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo
```

## Referencias

- Dataset: [UCI ML Repository - Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- Universidad Nacional de Tierra del Fuego (UNTDF) - Sistemas Inteligentes 2025

