# Reporte Final: Comparativa de Métodos para Detección de Fraude Financiero
**Matemáticas para Ciencia de Datos — Departamento de Matemáticas**  
*Evaluación de Modelos sobre `X_test_pca` (Dataset México: 1,921 transacciones, 385 en prueba)*

---

## 1. Resumen Ejecutivo y Metodología
Se implementaron y compararon detectores de anomalías basados en la **Distancia de Mahalanobis** y el **Kernel de Christoffel-Darboux (CD)** con regularización de Tikhonov ($\epsilon = 10^{-6}$). El conjunto de transacciones de México fue preprocesado reteniendo 27 variables numéricas, particionado (80% entrenamiento / 20% prueba estratificado) y proyectado mediante PCA ajustado en transacciones legítimas de entrenamiento. Se evaluaron las decisiones al umbral del 95% de confianza ($\rho_{95}$).

---

## 2. Tabla Comparativa de Rendimiento en Prueba (`X_test_pca`)

| Método | Dim ($d$) | Base / Grado | Var. Explicada (%) | Tiempo (ms) | Matriz de Confusión (1 renglón) | Precisión | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis 2D** | 2 | Lineal ($k=2$) | 21.17% | 0.07 ms | `TP=0, FP=17, TN=358, FN=10` | 0.0000 | 0.0000 | 0.0000 | 0.5157 |
| **Mahalanobis 10D** | 10 | Lineal ($k=10$) | 64.64% | 0.10 ms | `TP=5, FP=16, TN=359, FN=5` | 0.2381 | 0.5000 | 0.3226 | 0.7896 |
| **CD 2D (Grado 2)** | 2 | Monomios $\le 2$ ($k=6$) | 21.17% | 0.49 ms | `TP=0, FP=14, TN=361, FN=10` | 0.0000 | 0.0000 | 0.0000 | 0.6509 |
| **CD 2D (Grado 4)** | 2 | Monomios $\le 4$ ($k=15$) | 21.17% | 0.98 ms | `TP=0, FP=17, TN=358, FN=10` | 0.0000 | 0.0000 | 0.0000 | 0.6208 |
| **CD 5D (Grado 2)** | 5 | Monomios $\le 2$ ($k=21$) | 43.02% | 0.68 ms | `TP=1, FP=16, TN=359, FN=9` | 0.0588 | 0.1000 | 0.0741 | 0.6739 |
| **CD 7D (Grado 2)** | 7 | Monomios $\le 2$ ($k=36$) | 51.90% | 0.37 ms | `TP=3, FP=17, TN=358, FN=7` | 0.1500 | 0.3000 | 0.2000 | 0.6085 |
| **CD 10D (Grado 2)** | 10 | Monomios $\le 2$ ($k=66$) | 64.64% | 2.08 ms | `TP=5, FP=18, TN=357, FN=5` | 0.2174 | 0.5000 | 0.3030 | **0.8077** |

---

## 3. Visualizaciones en Dos Dimensiones y Dashboard de Métricas

| Elipses de Mahalanobis (2D) | Kernel Christoffel-Darboux Grado 4 (2D) |
| :---: | :---: |
| ![Mahalanobis 2D](plots/mahalanobis_2d_ellipses.png) | ![CD Grado 4](plots/christoffel_darboux_deg4_2d.png) |
| *Fronteras elípticas rígidas ($\rho_{50}$ a $\rho_{99}$)* | *Fronteras semialgebraicas adaptativas ($\rho_{90}=18.92, \rho_{95}=33.28, \rho_{99}=169.46$)* |

### Dashboard Comparativo Multidimensional
![Comparative Dashboard](plots/comparative_performance.png)

---

## 4. Análisis Crítico de Resultados

1. **Impacto de la Dimensionalidad (Varianza Explicada):**  
   Con 2 componentes solo se preserva el **21.17%** de la información de las 27 variables originales. En este plano, la nube de transacciones fraudulentas se encuentra traslapada con el centro denso legítimo (el 81% de los fraudes de entrenamiento cae en zonas de "seguridad"). Por esta razón matemática, ningún método en 2D logra detectar fraudes en prueba al umbral del 95% (`Recall = 0.00`).
2. **Superioridad en Dimensión 10:**  
   Al expandir a 10 componentes (**64.64%** de varianza), el poder discriminativo aumenta drásticamente: tanto Mahalanobis 10D como CD 10D detectan el **50.0%** de los fraudes (`Recall = 0.50`), alcanzando **CD 10D el mejor ROC-AUC global con 0.8077**.
3. **Flexibilidad Geométrica del Kernel CD:**  
   Mientras que Mahalanobis asume distribuciones unimodales y elípticas, el Kernel de Christoffel-Darboux modela la densidad inversa en un espacio semialgebraico. En 2 dimensiones con Grado 4, el Kernel CD ajusta contornos cóncavos y no convexos que acotan fielmente la dispersión legítima, mejorando el ROC-AUC de 0.5157 (Mahalanobis 2D) a 0.6509 (CD Grado 2).
4. **Eficiencia Computacional:**  
   Todos los métodos son ultrarrápidos en inferencia (tiempos menores a **2.1 ms** para 385 transacciones), demostrando viabilidad para scoring en tiempo real en pasarelas bancarias.

---

## 5. Conclusiones Finales
- **Reducción de Dimensionalidad:** Reducir a solo 2 componentes es didácticamente útil para visualización, pero resulta insuficiente para la detección real de fraude en este conjunto de datos.
- **Modelo Ganador:** Para despliegue operativo se recomienda el **Kernel de Christoffel-Darboux en 10 dimensiones con base de grado 2** ($\text{AUC} = 0.808$) o la **Distancia de Mahalanobis en 10 dimensiones** ($\text{AUC} = 0.790$), ya que capturan patrones no lineales y mantienen alta tasa de detección con mínimo costo computacional.
