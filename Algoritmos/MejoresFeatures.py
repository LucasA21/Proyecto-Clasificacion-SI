# ============================================
# Clasificación de sitios phishing con GaussianNB
# Esta técnica se asocia a las metaheurísticas de búsqueda global basadas en
# Monte Carlo, ya que utiliza muestreo aleatorio repetido para explorar
# subsets de variables en un espacio combinatorial muy grande.
# De este modo, aproxima una solución de alta calidad sin
# necesidad de evaluar todas las combinaciones posibles.
# ============================================

import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cargar_dataset_phishing

RANDOM_STATE = 42

# Crear carpeta de salida si no existe
output_dir = "resultados_phishing"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. Cargar dataset ARFF
# =========================
df = cargar_dataset_phishing()

print("Shape:", df.shape)
print(df['Result'].value_counts())

df.to_csv(os.path.join(output_dir, "dataset_preprocesado.csv"), index=False)

# =========================
# 2. Distribución de clases
# =========================
plt.figure()
df['Result'].value_counts().plot(kind="bar", title="Distribución de clases")
plt.savefig(os.path.join(output_dir, "distribucion_clases.png"))
plt.close()

# =========================
# 3. Separar datos
# =========================
X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================
# Random Search de subsets + Log en disco
# =========================
random.seed(RANDOM_STATE)

N_ITER = 500  # ajustar según tiempo disponible
K = 5  # tamaño del subset seleccionado

features = X_train.columns.tolist()
mejor_f1 = 0
mejor_subset = None

resultados = []
freq_optimos = Counter()  # contador de features más frecuentes en óptimos locales

log_path = os.path.join(output_dir, "log_random_search_subsets.csv")
freq_path = os.path.join(output_dir, "frecuencias_optimos.csv")

print("\n🔍 Iniciando Random Search de subsets...\n")

for i in range(N_ITER):
    subset = random.sample(features, K)

    model = GaussianNB()
    model.fit(X_train[subset], y_train)
    y_pred = model.predict(X_test[subset])
    f1 = f1_score(y_test, y_pred)

    resultados.append({
        "iteracion": i + 1,
        "subset": subset,
        "f1_score": f1
    })

    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_subset = subset

        # actualizar el contador de features que aparecen en óptimos locales
        freq_optimos.update(subset)

        print(f"⭐ Óptimo local en iteración {i + 1}/{N_ITER}")
        print(f"   F1-score: {f1:.4f}")
        print(f"   Variables: {subset}\n")
        print("📈 Frecuencias actuales de features más frecuentes:")
        print(freq_optimos.most_common(10))
        print("\n")

        # Guardado incremental del log
        pd.DataFrame(resultados).sort_values(
            by="f1_score", ascending=False
        ).to_csv(log_path, index=False)

        # Guardado incremental del ranking de features
        pd.DataFrame(freq_optimos.most_common()).to_csv(freq_path, index=False,
                                                        header=["feature", "frecuencia"])

print("\n✅ Búsqueda finalizada")
print(f"🏆 Mejor F1-score global: {mejor_f1:.4f}")
print(f"🏆 Mejor subset encontrado:")
print(mejor_subset)

# Guardado final consolidado
df_resultados = pd.DataFrame(resultados).sort_values(by="f1_score", ascending=False)
df_resultados.to_csv(log_path, index=False)

pd.DataFrame(freq_optimos.most_common()).to_csv(freq_path, index=False,
                                                header=["feature", "frecuencia"])

print(f"\n📄 Resultados guardados en:")
print(f" - {log_path}")
print(f" - {freq_path}")

# Obtener las 10 columnas más frecuentes en óptimos locales
top10_features = [feature for feature, _ in freq_optimos.most_common(K)]

print(f"\n🔎 Top {K} features más frecuentes en óptimos locales:")
print(top10_features)

# Entrenar un modelo con estas columnas
model_top10 = GaussianNB()
model_top10.fit(X_train[top10_features], y_train)
y_pred_top10 = model_top10.predict(X_test[top10_features])

# Evaluación
from sklearn.metrics import classification_report, f1_score, confusion_matrix

f1_top10 = f1_score(y_test, y_pred_top10)

print(f"\n📊 Resultados usando las {K} features más frecuentes:")
print(classification_report(y_test, y_pred_top10))
print(f"F1-score Top10 = {f1_top10:.4f}")

# Matriz de confusión
cm_top10 = confusion_matrix(y_test, y_pred_top10)
print(f"Matriz de confusión (Top-{K}):")
print(cm_top10)

# Comparación rápida
print("\n📌 Comparación Final")
print(f"Mejor F1-score Random Search: {mejor_f1:.4f}")
print(f"F1-score con Top10 frecuentes: {f1_top10:.4f}")

