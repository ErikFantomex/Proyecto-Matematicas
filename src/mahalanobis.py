"""
Módulo para la Primera Propuesta: Detector de Fraude basado en la Distancia de Mahalanobis.
"""

import numpy as np
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


class MahalanobisDetector:
    def __init__(self, n_components=2, centered_at_origin=True):
        """
        Detector de anomalías basado en la distancia de Mahalanobis.
        :param n_components: Número de componentes de PCA a utilizar (e.g. 2 o 10).
        :param centered_at_origin: Si es True, usa d_M(x)^2 = x^T \Sigma^-1 x como en el PDF;
                                   si es False, usa (x - mu)^T \Sigma^-1 (x - mu).
        """
        self.n_components = n_components
        self.centered_at_origin = centered_at_origin
        self.cov = None
        self.inv_cov = None
        self.mean = None
        self.train_distances = None
        self.radii = {}

    def fit(self, X_train_legit):
        """
        Ajusta la matriz de covarianza sobre las transacciones legítimas de entrenamiento.
        """
        X = X_train_legit[:, :self.n_components]
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X, rowvar=False)
        # Pseudoinversa regularizada para garantizar estabilidad numérica
        self.inv_cov = np.linalg.pinv(self.cov)
        
        # Calcular distancias en entrenamiento
        self.train_distances = self.compute_distance(X)
        return self

    def compute_distance(self, X):
        """
        Calcula d_M(x) para cada vector en X.
        """
        X_sub = X[:, :self.n_components]
        if self.centered_at_origin:
            diff = X_sub
        else:
            diff = X_sub - self.mean
            
        # d_M(x)^2 = diff @ inv_cov @ diff.T
        dist_sq = np.sum((diff @ self.inv_cov) * diff, axis=1)
        # Evitar valores negativos por precisión numérica
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)

    def find_radius(self, n_percent):
        """
        Encuentra el radio rho_n tal que exactamente n% de los datos legítimos
        se encuentre dentro de B_{rho_n}(0) = {x : d_M(x) <= rho_n},
        pero no contenga (n + 1)%.
        """
        N = len(self.train_distances)
        sorted_dists = np.sort(self.train_distances)
        
        # Índice que delimita el n% inferior
        idx = int(np.floor(n_percent * N / 100.0))
        idx = min(max(idx, 0), N - 1)
        rho_n = sorted_dists[idx]
        self.radii[n_percent] = rho_n
        return rho_n

    def compute_risk_zones(self, X_fraud):
        """
        Evalúa las transacciones fraudulentas en los distintos radios de seguridad:
        - Seguridad Extrema: d <= rho_50
        - Seguridad Alta: rho_50 < d <= rho_75
        - Seguridad Baja: rho_75 < d <= rho_90
        - Peligro: d > rho_90 (y subnivel de peligro extremo d > rho_95)
        """
        for p in [50, 75, 90, 95, 99]:
            if p not in self.radii:
                self.find_radius(p)
                
        d_fraud = self.compute_distance(X_fraud)
        total_frauds = len(d_fraud)
        
        zones = {
            'extrema': np.sum(d_fraud <= self.radii[50]),
            'alta': np.sum((d_fraud > self.radii[50]) & (d_fraud <= self.radii[75])),
            'baja': np.sum((d_fraud > self.radii[75]) & (d_fraud <= self.radii[90])),
            'peligro': np.sum(d_fraud > self.radii[90]),
            'peligro_extremo': np.sum(d_fraud > self.radii[95]),
            'total_frauds': total_frauds
        }
        return zones

    def evaluate(self, X_test, y_test, threshold_percentile=95):
        """
        Evalúa el detector en el conjunto de prueba considerando como fraude
        aquellas transacciones con d_M(x) > rho_n.
        Retorna tiempo de ejecución, matriz de confusión (en un renglón) y métricas.
        """
        t0 = time.perf_counter()
        rho = self.radii.get(threshold_percentile, self.find_radius(threshold_percentile))
        
        d_test = self.compute_distance(X_test)
        y_pred = (d_test > rho).astype(int)
        exec_time_ms = (time.perf_counter() - t0) * 1000
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, d_test)
        
        cm_line = f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
        
        return {
            'exec_time_ms': exec_time_ms,
            'threshold_percentile': threshold_percentile,
            'rho': rho,
            'cm': cm,
            'cm_line': cm_line,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'distances': d_test,
            'predictions': y_pred
        }
