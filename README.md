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

