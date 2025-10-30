# ============================================
# Clasificación de sitios phishing con GaussianNB
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
from scipy.io import arff

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cargar_dataset_phishing

RANDOM_STATE = 13
#42 >>> F1         0.433842    0.645445
#73 >>> F1         0.420985    0.630531
#95 >>> F1         0.436271    0.621727
#13 >>> F1         0.449749    0.656522

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
# 4. Escalado de features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

# =========================
# 5. Modelo base
# =========================
model_base = GaussianNB()
model_base.fit(X_train_scaled, y_train)
y_pred_base = model_base.predict(X_test_scaled)

# =========================
# 6. Optimización de Hiperparámetros
# =========================
param_grid = {"var_smoothing": np.logspace(-12, -6, 7)}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    GaussianNB(), param_grid,
    scoring="f1", cv=cv, n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
joblib.dump(best_model, os.path.join(output_dir, "modelo_mejorado.joblib"))

y_pred_best = best_model.predict(X_test_scaled)

# =========================
# 7. Matriz de confusión
# =========================
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Modelo Optimizado")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig(os.path.join(output_dir, "matriz_confusion.png"))
plt.close()

# =========================
# 8. Comparación de métricas
# =========================
def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

results = pd.DataFrame({
    "Base": metrics(y_test, y_pred_base),
    "Optimizado": metrics(y_test, y_pred_best)
})

print("\n--- Comparación Base vs Optimizado ---")
print(results)

# Exportar métricas
results.to_csv(os.path.join(output_dir, "comparacion_metricas.csv"))
