"""
Módulo para la Segunda Propuesta: Detector de Fraude basado en el Kernel de Christoffel-Darboux (CD).
"""

import numpy as np
import time
from itertools import combinations
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


class ChristoffelDarbouxDetector:
    def __init__(self, n_components=2, degree=2, epsilon=1e-6):
        """
        Detector de anomalías basado en el Kernel de Christoffel-Darboux.
        :param n_components: Número de componentes de PCA a utilizar (2, 5, 7, 10).
        :param degree: Grado del polinomio base (2 para cualquier dimensión, 4 para 2 dimensiones).
        :param epsilon: Parámetro de regularización de Tikhonov (ridge) M_reg = M_n + epsilon * I.
        """
        self.n_components = n_components
        self.degree = degree
        self.epsilon = epsilon
        
        self.M_n = None
        self.M_reg = None
        self.M_inv = None
        self.train_scores = None
        self.radii = {}
        self.feature_dim = None

    def transform_features(self, X):
        """
        Mapea el vector x en R^d a la base polinomial Phi(x).
        """
        X_sub = X[:, :self.n_components]
        N, d = X_sub.shape
        
        if d == 2 and self.degree == 2:
            # Base canónica bi-variada en R^6: (1, x1, x2, x1^2, x1*x2, x2^2)
            x1 = X_sub[:, 0]
            x2 = X_sub[:, 1]
            Phi = np.column_stack([
                np.ones(N),
                x1, x2,
                x1**2, x1 * x2, x2**2
            ])
            return Phi
            
        elif d == 2 and self.degree == 4:
            # Base canónica bi-variada en R^15: polinomios de grado <= 4
            x1 = X_sub[:, 0]
            x2 = X_sub[:, 1]
            Phi = np.column_stack([
                # Grado 0
                np.ones(N),
                # Grado 1
                x1, x2,
                # Grado 2
                x1**2, x1 * x2, x2**2,
                # Grado 3
                x1**3, (x1**2) * x2, x1 * (x2**2), x2**3,
                # Grado 4
                x1**4, (x1**3) * x2, (x1**2) * (x2**2), x1 * (x2**3), x2**4
            ])
            return Phi
            
        elif self.degree == 2:
            # Base general de grado 2 para d dimensiones (R^k con k = 1 + 2d + d(d-1)/2)
            cols = [np.ones((N, 1)), X_sub, X_sub**2]
            
            # Términos cruzados xi * xj para i < j
            cross_terms = []
            for i, j in combinations(range(d), 2):
                cross_terms.append((X_sub[:, i] * X_sub[:, j])[:, None])
                
            if cross_terms:
                cols.append(np.hstack(cross_terms))
                
            return np.hstack(cols)
            
        else:
            raise NotImplementedError(f"Grado {self.degree} para dimensión {d} no implementado directamente.")

    def fit(self, X_train_legit):
        """
        Calcula la matriz de momentos empírica M_n = (1/N) D^T D y su inversa regularizada M_reg^-1.
        """
        D = self.transform_features(X_train_legit)
        N, k = D.shape
        self.feature_dim = k
        
        # Matriz de momentos empírica M_n in R^{k x k}
        self.M_n = (D.T @ D) / N
        
        # Regularización de Tikhonov (ridge)
        self.M_reg = self.M_n + self.epsilon * np.eye(k)
        
        # Inversa
        self.M_inv = np.linalg.inv(self.M_reg)
        
        # Calcular el polinomio de Christoffel-Darboux en entrenamiento
        self.train_scores = self.compute_pcd(X_train_legit)
        return self

    def compute_pcd(self, X):
        """
        Calcula el polinomio p_CD(x) = Phi(x)^T M_reg^-1 Phi(x) para cada muestra.
        """
        Phi = self.transform_features(X)
        # Vectorizado: diag(Phi @ M_inv @ Phi^T) = sum((Phi @ M_inv) * Phi, axis=1)
        p_cd = np.sum((Phi @ self.M_inv) * Phi, axis=1)
        return p_cd

    def find_radius(self, n_percent):
        """
        Encuentra el radio de nivel rho_n tal que exactamente el n% de transacciones legítimas
        satisfaga p_CD(x) <= rho_n, sin contener al (n+1)%.
        """
        N = len(self.train_scores)
        sorted_scores = np.sort(self.train_scores)
        idx = int(np.floor(n_percent * N / 100.0))
        idx = min(max(idx, 0), N - 1)
        rho_n = sorted_scores[idx]
        self.radii[n_percent] = rho_n
        return rho_n

    def evaluate(self, X_test, y_test, threshold_percentile=95):
        """
        Evalúa el detector en el conjunto de prueba considerando anomalía/fraude si p_CD(x) > rho_n.
        """
        t0 = time.perf_counter()
        rho = self.radii.get(threshold_percentile, self.find_radius(threshold_percentile))
        
        p_test = self.compute_pcd(X_test)
        y_pred = (p_test > rho).astype(int)
        exec_time_ms = (time.perf_counter() - t0) * 1000
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, p_test)
        
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
            'scores': p_test,
            'predictions': y_pred,
            'feature_dim': self.feature_dim
        }
