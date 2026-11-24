import json

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.split('\n')]
    })

def add_code(code):
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [line + '\n' for line in code.split('\n')]
    })

# Title
add_md('''# Proyecto Final: Matemáticas para Ciencia de Datos
## Detección de Fraude Financiero mediante Distancia de Mahalanobis y Kernel de Christoffel-Darboux
**Departamento de Matemáticas — Octubre 2026**

Este notebook implementa el flujo de trabajo completo del proyecto:
1. **Adquisición y preparación del conjunto de datos:** Filtrado de transacciones de México, estandarización y reducción con PCA (10 componentes).
2. **Primera propuesta:** Detector basado en la Distancia de Mahalanobis en 2D y 10D, cálculo de radios de seguridad $\\rho_n$ y elipses de nivel.
3. **Segunda propuesta:** Detector basado en el Kernel de Christoffel-Darboux (CD) usando bases polinomiales de Grado 2 y Grado 4 (en 2D, 5D, 7D y 10D) con regularización de Tikhonov.
4. **Evaluación comparativa:** Tiempos de ejecución, varianza explicada, matrices de confusión en un renglón, métricas (Precisión, Recall, F1, ROC-AUC) y visualizaciones.''')

# Imports
add_code('''import os
import glob
import time
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Configuración estética
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
print("Librerías importadas correctamente.")''')

# Section 1
add_md('''## 1. Adquisición y Preparación del Conjunto de Datos

El dataset proviene de Kaggle: `mohankrishnathalla/credit-card-fraud-detection-dataset-2026`.
- Filtramos únicamente las transacciones de México (`merchant_country == 'Mexico'`).
- Consideramos solo las columnas numéricas como características predictoras y la etiqueta binaria `is_fraud`.
- Separamos 80% entrenamiento y 20% prueba de forma estratificada.
- Ajustamos `StandardScaler` y `PCA(n_components=10)` **exclusivamente** sobre el conjunto de entrenamiento.''')

add_code('''# 1. Carga del conjunto de datos
cache_base = os.path.expanduser(r"~/.cache/kagglehub/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026")
matches = glob.glob(os.path.join(cache_base, "**", "credit_card_fraud_detection_2026.csv"), recursive=True)
if matches:
    csv_path = matches[0]
else:
    path = kagglehub.dataset_download("mohankrishnathalla/credit-card-fraud-detection-dataset-2026")
    csv_path = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)[0]

df = pd.read_csv(csv_path)
df_mexico = df[df["merchant_country"] == "Mexico"].copy()

# Columnas numéricas predictoras (excluyendo is_fraud)
num_cols = [c for c in df_mexico.select_dtypes(include=[np.number]).columns if c != "is_fraud"]

print(f"Total transacciones en México: {len(df_mexico)}")
print(f"Transacciones legítimas: {(df_mexico['is_fraud'] == 0).sum()}, Fraudes: {(df_mexico['is_fraud'] == 1).sum()}")
print(f"Número de variables numéricas predictoras: {len(num_cols)}")''')

add_code('''# 2. Separación Entrenamiento (80%) y Prueba (20%)
train_df, test_df = train_test_split(
    df_mexico, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_mexico["is_fraud"]
)

X_train_raw = train_df[num_cols].values
y_train = train_df["is_fraud"].values

X_test_raw = test_df[num_cols].values
y_test = test_df["is_fraud"].values

# 3. Estandarización y PCA ajustados solo en Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Transacciones legítimas de entrenamiento
X_train_legit = X_train_pca[y_train == 0]
X_train_fraud = X_train_pca[y_train == 1]

cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
print(f"Train: {len(X_train_pca)} muestras (Legítimas: {len(X_train_legit)}, Fraude: {len(X_train_fraud)})")
print(f"Test:  {len(X_test_pca)} muestras (Legítimas: {(y_test==0).sum()}, Fraude: {(y_test==1).sum()})")
print(f"Varianza explicada con 2 componentes:  {cum_var[1]:.2f}%")
print(f"Varianza explicada con 10 componentes: {cum_var[9]:.2f}%")''')

# Section 2
add_md('''## 2. Primera Propuesta: Distancia de Mahalanobis

La distancia de Mahalanobis centrada en el origen para una transacción $x$ se define como:
$$d_M(x)^2 = x^T \\Sigma^{-1} x$$
donde $\\Sigma$ es la matriz de covarianza de las transacciones legítimas de entrenamiento (`X_train_legit`).

Encontramos los radios $\\rho_n$ correspondientes al $n\\%$ de transacciones legítimas ($n \\in \\{50, 75, 90, 95, 99\\}$) y definimos zonas de riesgo para clasificar transacciones legítimas vs fraudulentas.''')

add_code('''class MahalanobisDetector:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.cov = None
        self.inv_cov = None
        self.train_distances = None
        self.radii = {}

    def fit(self, X_train_legit):
        X = X_train_legit[:, :self.n_components]
        self.cov = np.cov(X, rowvar=False)
        self.inv_cov = np.linalg.pinv(self.cov)
        self.train_distances = self.compute_distance(X)
        return self

    def compute_distance(self, X):
        X_sub = X[:, :self.n_components]
        dist_sq = np.sum((X_sub @ self.inv_cov) * X_sub, axis=1)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def find_radius(self, n_percent):
        N = len(self.train_distances)
        sorted_dists = np.sort(self.train_distances)
        idx = int(np.floor(n_percent * N / 100.0))
        idx = min(max(idx, 0), N - 1)
        rho_n = sorted_dists[idx]
        self.radii[n_percent] = rho_n
        return rho_n

    def evaluate(self, X_test, y_test, threshold_percentile=95):
        t0 = time.perf_counter()
        rho = self.radii.get(threshold_percentile, self.find_radius(threshold_percentile))
        d_test = self.compute_distance(X_test)
        y_pred = (d_test > rho).astype(int)
        exec_time = (time.perf_counter() - t0) * 1000
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, d_test)
        
        return {
            'exec_time_ms': exec_time,
            'cm_line': f"TP={tp}, FP={fp}, TN={tn}, FN={fn}",
            'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'distances': d_test
        }

# Ajustar Mahalanobis 2D y 10D
maha_2d = MahalanobisDetector(n_components=2).fit(X_train_legit)
maha_10d = MahalanobisDetector(n_components=10).fit(X_train_legit)

print("Radios Mahalanobis 2D:")
for p in [50, 75, 90, 95, 99]:
    print(f"  rho_{p}: {maha_2d.find_radius(p):.4f}")

print("\nRadios Mahalanobis 10D:")
for p in [50, 75, 90, 95, 99]:
    print(f"  rho_{p}: {maha_10d.find_radius(p):.4f}")''')

add_code('''# Gráfica de Elipses de Mahalanobis (2D)
fig, ax = plt.subplots(figsize=(9, 7))

all_pts = np.vstack([X_train_legit[:, :2], X_train_fraud[:, :2]])
x_min, x_max = all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1
y_min, y_max = all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
d_grid = maha_2d.compute_distance(grid_pts).reshape(xx.shape)

ax.scatter(X_train_legit[:, 0], X_train_legit[:, 1], c="#1f77b4", alpha=0.4, s=18, label="Legítimas")
ax.scatter(X_train_fraud[:, 0], X_train_fraud[:, 1], c="#d62728", alpha=0.9, s=32, edgecolors="black", label="Fraude")

percentiles = [50, 75, 90, 95, 99]
radii = [maha_2d.find_radius(p) for p in percentiles]
colors = ["#2ca02c", "#17becf", "#ff7f0e", "#9467bd", "#8c564b"]

cs = ax.contour(xx, yy, d_grid, levels=radii, colors=colors, linewidths=1.8)
fmt = {r: f"{p}% ({r:.2f})" for r, p in zip(radii, percentiles)}
ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=8)

ax.set_title("Fronteras de Decisión - Distancia de Mahalanobis (2D)", fontsize=13, fontweight="bold")
ax.set_xlabel("PC1 (Estandarizada)")
ax.set_ylabel("PC2 (Estandarizada)")
ax.legend(loc="upper right")
plt.show()''')

# Section 3
add_md('''## 3. Segunda Propuesta: Kernel de Christoffel-Darboux (CD)

Para una transacción $x = (x_1, x_2) \\in \\mathbb{R}^2$, consideramos la base de monomios bi-variados:
- **Grado 2 (en $\\mathbb{R}^6$):** $\\Phi(x) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)^T$
- **Grado 4 (en $\\mathbb{R}^{15}$):** Monomios de grado $\\le 4$.

La matriz de momentos empírica es $M_n = \\frac{1}{N} D^T D \\in \\mathbb{R}^{k \\times k}$. Aplicamos regularización de Tikhonov:
$$M_{\\text{reg}} = M_n + \\epsilon I, \\quad \\epsilon = 10^{-6}$$
El polinomio de Christoffel-Darboux se evalúa como:
$$p_{CD}(x) = \\Phi(x)^T M_{\\text{reg}}^{-1} \\Phi(x)$$
y las fronteras de decisión corresponden a los subconjuntos de nivel $S_\\rho = \\{x : p_{CD}(x) \\le \\rho\\}$.''')

add_code('''class ChristoffelDarbouxDetector:
    def __init__(self, n_components=2, degree=2, epsilon=1e-6):
        self.n_components = n_components
        self.degree = degree
        self.epsilon = epsilon
        self.M_inv = None
        self.train_scores = None
        self.radii = {}

    def transform_features(self, X):
        X_sub = X[:, :self.n_components]
        N, d = X_sub.shape
        if d == 2 and self.degree == 2:
            x1, x2 = X_sub[:, 0], X_sub[:, 1]
            return np.column_stack([np.ones(N), x1, x2, x1**2, x1*x2, x2**2])
        elif d == 2 and self.degree == 4:
            x1, x2 = X_sub[:, 0], X_sub[:, 1]
            return np.column_stack([
                np.ones(N),
                x1, x2,
                x1**2, x1*x2, x2**2,
                x1**3, (x1**2)*x2, x1*(x2**2), x2**3,
                x1**4, (x1**3)*x2, (x1**2)*(x2**2), x1*(x2**3), x2**4
            ])
        elif self.degree == 2:
            cols = [np.ones((N, 1)), X_sub, X_sub**2]
            for i, j in combinations(range(d), 2):
                cols.append((X_sub[:, i] * X_sub[:, j])[:, None])
            return np.hstack(cols)
        raise NotImplementedError()

    def fit(self, X_train_legit):
        D = self.transform_features(X_train_legit)
        N, k = D.shape
        M_n = (D.T @ D) / N
        M_reg = M_n + self.epsilon * np.eye(k)
        self.M_inv = np.linalg.inv(M_reg)
        self.train_scores = self.compute_pcd(X_train_legit)
        return self

    def compute_pcd(self, X):
        Phi = self.transform_features(X)
        return np.sum((Phi @ self.M_inv) * Phi, axis=1)

    def find_radius(self, n_percent):
        N = len(self.train_scores)
        sorted_s = np.sort(self.train_scores)
        idx = min(max(int(np.floor(n_percent * N / 100.0)), 0), N - 1)
        rho_n = sorted_s[idx]
        self.radii[n_percent] = rho_n
        return rho_n

    def evaluate(self, X_test, y_test, threshold_percentile=95):
        t0 = time.perf_counter()
        rho = self.radii.get(threshold_percentile, self.find_radius(threshold_percentile))
        p_test = self.compute_pcd(X_test)
        y_pred = (p_test > rho).astype(int)
        exec_time = (time.perf_counter() - t0) * 1000
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'exec_time_ms': exec_time,
            'cm_line': f"TP={tp}, FP={fp}, TN={tn}, FN={fn}",
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, p_test),
            'scores': p_test
        }''')

add_code('''# Gráfica Réplica de Portada: Kernel CD Grado 4 (2D)
cd_deg4 = ChristoffelDarbouxDetector(n_components=2, degree=4).fit(X_train_legit)

fig, ax = plt.subplots(figsize=(9, 7))
x_min, x_max = -2.5, 14.5
y_min, y_max = -3.5, 7.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
p_grid = cd_deg4.compute_pcd(grid_pts).reshape(xx.shape)

ax.scatter(X_train_legit[:, 0], X_train_legit[:, 1], c="#3b82f6", alpha=0.35, s=16, label="Transacciones Legítimas")
ax.scatter(X_train_fraud[:, 0], X_train_fraud[:, 1], c="#b91c1c", alpha=0.9, s=32, edgecolors="black", label="Fraude")

levels = [90, 95, 99]
radii = [cd_deg4.find_radius(p) for p in levels]
colors = ["#0ea5e9", "#6366f1", "#4338ca"]

cs = ax.contour(xx, yy, p_grid, levels=radii, colors=colors, linewidths=1.8)
fmt = {r: f"{r:.3f}" for r in radii}
ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=8)

lines = [plt.Line2D([0], [0], color=c, lw=1.8) for c in colors]
labels = [f"Frontera Densidad {p}%" for p in levels]
handles, orig_labels = ax.get_legend_handles_labels()
ax.legend(handles + lines, orig_labels + labels, loc="upper right")

ax.set_title("Fronteras de Decisión (3 Niveles del Kernel CD) - Grado 4", fontsize=13, fontweight="bold")
ax.set_xlabel("PC1 (Estandarizada)")
ax.set_ylabel("PC2 (Estandarizada)")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.show()''')

# Section 4
add_md('''## 4. Comparativa de Todos los Modelos sobre `X_test_pca`

Evaluamos todos los detectores (Mahalanobis 2D/10D y Kernel CD en 2D, 5D, 7D y 10D) al umbral del 95%.''')

add_code('''models = [
    ('Mahalanobis 2D', maha_2d, 2, cum_var[1]),
    ('Mahalanobis 10D', maha_10d, 10, cum_var[9]),
    ('CD 2D (Deg 2)', ChristoffelDarbouxDetector(2, 2).fit(X_train_legit), 2, cum_var[1]),
    ('CD 2D (Deg 4)', cd_deg4, 2, cum_var[1]),
    ('CD 5D (Deg 2)', ChristoffelDarbouxDetector(5, 2).fit(X_train_legit), 5, cum_var[4]),
    ('CD 7D (Deg 2)', ChristoffelDarbouxDetector(7, 2).fit(X_train_legit), 7, cum_var[6]),
    ('CD 10D (Deg 2)', ChristoffelDarbouxDetector(10, 2).fit(X_train_legit), 10, cum_var[9]),
]

results = []
for name, detector, dims, v_exp in models:
    res = detector.evaluate(X_test_pca, y_test, threshold_percentile=95)
    results.append({
        'Método': name,
        'Dim': dims,
        'Var. Explicada (%)': f"{v_exp:.2f}%",
        'Tiempo (ms)': f"{res['exec_time_ms']:.2f} ms",
        'Matriz Confusión (1 renglón)': res['cm_line'],
        'Precisión': f"{res['precision']:.4f}",
        'Recall': f"{res['recall']:.4f}",
        'F1-Score': f"{res['f1']:.4f}",
        'ROC-AUC': f"{res['roc_auc']:.4f}",
    })

pd.DataFrame(results)''')

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbformat': 4,
            'nbformat_minor': 4
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('Proyecto_Final_MCD.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook Proyecto_Final_MCD.ipynb generado exitosamente.')
