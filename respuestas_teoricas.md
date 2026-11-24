# Respuestas Teóricas y Matemáticas Detalladas
## Proyecto Final: Matemáticas para Ciencia de Datos
### Detección de Fraude Financiero mediante Distancia de Mahalanobis y Kernel de Christoffel-Darboux
**Departamento de Matemáticas**

---

### Pregunta 1: Importancia de la Estandarización de Datos
> *Primero se estandariza —¿por qué es importante esta parte?—:*
> ```python
> from sklearn.preprocessing import StandardScaler
> scaler = StandardScaler()
> X_train_scaled = scaler.fit_transform(X_train)
> ```

**Respuesta:**

1. **Heterogeneidad de Escalas y Unidades de Medida:**  
   El conjunto de datos contiene atributos con órdenes de magnitud diametralmente opuestos. Por ejemplo:
   - `transaction_amount` (montos monetarios en cientos o miles).
   - `account_age_days` (antigüedad de cuentas que alcanza miles de días).
   - `distance_from_home_km` (distancias en decenas o centenas).
   - En contraste, variables como `is_weekend`, `card_present` o `used_pin` son binarias ($\in \{0, 1\}$).

2. **Sensibilidad del Análisis de Componentes Principales (PCA):**  
   PCA identifica direcciones ortogonales que maximizan la varianza proyectada $\text{Var}(X v)$. Si los datos no se estandarizan, las variables con valores numéricos grandes presentarán varianzas muestrales artificialmente gigantescas en comparación con las variables en escalas pequeñas, sin que ello signifique mayor poder predictivo. Como resultado, las primeras componentes principales quedarían alineadas exclusivamente con los ejes de mayor escala numérica, oscureciendo las correlaciones intrínsecas y la estructura latente multivariada.

3. **Equivalencia con la Matriz de Correlación:**  
   Al transformar cada característica mediante $z_j = \frac{x_j - \mu_j}{\sigma_j}$, cada columna queda con media 0 y varianza unitaria ($\sigma^2 = 1$). Matemáticamente, la matriz de covarianza de los datos estandarizados $\frac{1}{N-1} Z^T Z$ coincide de forma exacta con la **matriz de correlación** del conjunto original, garantizando que cada atributo contribuya en igualdad de condiciones a la descomposición espectral.

---

### Pregunta 2: Interpretación de las 10 Componentes mediante la SVD
> *En términos de las matrices de la SVD, ¿quiénes son las 10 componentes con las que nos vamos a quedar?*

**Respuesta:**

Sea $X_{\text{scaled}} \in \mathbb{R}^{N \times p}$ la matriz centrada y estandarizada de entrenamiento ($N = 1536$ transacciones, $p = 27$ variables numéricas). La descomposición en valores singulares (SVD) de $X_{\text{scaled}}$ se expresa como:

$$X_{\text{scaled}} = U \Sigma V^T$$

donde:
- $U \in \mathbb{R}^{N \times N}$ es una matriz ortogonal cuyas columnas $u_i$ son los vectores singulares izquierdos (autovectores ortonormales de $X X^T$).
- $\Sigma \in \mathbb{R}^{N \times p}$ es una matriz diagonal rectangular que contiene los valores singulares ordenados en forma decreciente: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(N,p)} \ge 0$.
- $V \in \mathbb{R}^{p \times p}$ es una matriz ortogonal cuyas columnas $v_j$ son los vectores singulares derechos (autovectores ortonormales de la matriz de covarianza muestral $S = \frac{1}{N-1} X^T X$, asociados a los autovalores $\lambda_j = \frac{\sigma_j^2}{N-1}$).

Las **10 componentes con las que nos quedamos** corresponden a:

1. **Ejes Principales (Loadings / Direcciones ortogonales):**  
   Son las primeras 10 columnas de la matriz $V$, representadas por la submatriz:
   $$V_{10} = [v_1, v_2, \dots, v_{10}] \in \mathbb{R}^{p \times 10}$$
   asociadas a los 10 mayores valores singulares $\sigma_1, \dots, \sigma_{10}$.

2. **Coordenadas Proyectadas (Scores / $X_{\text{train\_pca}}$):**  
   Son las primeras 10 columnas de la proyección $Z = X_{\text{scaled}} V$, que por la descomposición SVD equivale a:
   $$X_{\text{train\_pca}} = X_{\text{scaled}} V_{10} = U_{10} \Sigma_{10} \in \mathbb{R}^{N \times 10}$$
   donde $U_{10} \in \mathbb{R}^{N \times 10}$ contiene los primeros 10 vectores singulares izquierdos y $\Sigma_{10} = \text{diag}(\sigma_1, \dots, \sigma_{10}) \in \mathbb{R}^{10 \times 10}$. Cada renglón de esta matriz representa la posición de una transacción en el subespacio óptimo de dimensión 10.

---

### Pregunta 3: Cálculo Matemático en la Transformación del Conjunto de Prueba
> *Para dejar listo el conjunto de pruebas, aplica `scaler.transform` y luego `pca.transform` a `X_test`. En esta parte no se hace de cero la reducción de dimensionalidad, sino que se aplican las funciones aprendidas en el conjunto de entrenamiento tanto en el escalado estándar como en el PCA de 10 componentes. Matemáticamente, ¿qué estamos calculando exactamente? Llamaremos a este conjunto `X_test_pca`.*

**Respuesta:**

Matemáticamente estamos calculando la **proyección ortogonal de las nuevas observaciones sobre el subespacio de dimensión 10 determinado por los ejes principales de entrenamiento**, luego de aplicarles una traslación y cambio de escala inducidos por la distribución de entrenamiento:

1. **Estandarización Exógena (`scaler.transform`):**  
   Para cada muestra de prueba $x_{\text{test}} \in \mathbb{R}^p$, calculamos:
   $$\tilde{x}_{\text{test}} = S_{\text{train}}^{-1} (x_{\text{test}} - \mu_{\text{train}})$$
   donde $\mu_{\text{train}} = \frac{1}{N} \sum_{i=1}^N x_i \in \mathbb{R}^p$ y $S_{\text{train}} = \text{diag}(\sigma_{1,\text{train}}, \dots, \sigma_{p,\text{train}})$ son la media y desviación estándar estimadas **únicamente con los datos de entrenamiento**. Esto evita *data leakage* y respeta la causalidad temporal del mundo real.

2. **Proyección Lineal Ortogonal (`pca.transform`):**  
   Calculamos el producto punto de $\tilde{x}_{\text{test}}$ con la base ortonormal $V_{10}$ aprendida:
   $$z_{\text{test}} = \tilde{x}_{\text{test}} V_{10} \in \mathbb{R}^{10}$$
   En términos matriciales compactos para todo el bloque de prueba $X_{\text{test}} \in \mathbb{R}^{M \times p}$:
   $$X_{\text{test\_pca}} = \left( (X_{\text{test}} - \mathbf{1}_M \mu_{\text{train}}^T) S_{\text{train}}^{-1} \right) V_{10} \in \mathbb{R}^{M \times 10}$$
   Expresamos las transacciones no vistas en el sistema coordenado generado por los 10 modos de variación dominante del conjunto de entrenamiento.

---

### Pregunta 4: Dimensión de Cálculo de la Distancia de Mahalanobis
> *¿En qué dimensión se calcula esta distancia?*

**Respuesta:**

La distancia de Mahalanobis $d_M(x)^2 = x^T \Sigma^{-1} x$ se calcula en la **dimensión del espacio coordenado donde se define el vector $x$ y la matriz de covarianza $\Sigma$**:
- En el modelo multivariado de 10 componentes principales, se calcula en **dimensión 10** ($\mathbb{R}^{10}$), con $\Sigma \in \mathbb{R}^{10 \times 10}$.
- En el ejercicio didáctico bidimensional, se calcula en **dimensión 2** ($\mathbb{R}^2$), con $\Sigma \in \mathbb{R}^{2 \times 2}$.

---

### Pregunta 5: Eficiencia y Varianza Explicada con Dos Componentes
> *¿Pierde eficiencia tu método si solo consideramos dos componentes? ¿Cuál es la varianza explicada de tus datos para dos componentes?*

**Respuesta:**

1. **Varianza Explicada en 2 Componentes:**  
   Las dos primeras componentes principales explican únicamente el **21.17%** de la varianza total de los datos ($\text{PC}_1 = 12.94\%$, $\text{PC}_2 = 8.23\%$). Por el contrario, 10 componentes capturan el **64.64%** de la varianza.

2. **Pérdida Crítica de Eficiencia y Separabilidad:**  
   **Sí, el método pierde drásticamente su eficiencia:**
   - Al comprimir 27 variables a 2 dimensiones, se pierde casi el 79% de la información geométrica. Los patrones fraudulentos quedan colapsados en el interior de la nube de transacciones legítimas (el 81.0% de los fraudes de entrenamiento cae dentro de los radios de seguridad extrema $\rho_{50}$ y alta $\rho_{75}$).
   - En el conjunto de prueba (`X_test_pca`), el detector con 2 componentes obtiene un Recall de **0.000** (0 fraudes detectados al umbral del 95%) y un ROC-AUC de apenas **0.516** (equivalente al azar).
   - Al utilizar las 10 componentes, el ROC-AUC se eleva a **0.790**, alcanzando un Recall de **0.500** (5 de 10 fraudes detectados) y un F1 de **0.323**. Las dimensiones 3 a 10 contienen las señales discriminantes clave de comportamiento anómalo.

---

### Pregunta 6: Relación entre la Matriz de Momentos y la Distancia de Mahalanobis
> *donde $D$ es la matriz en $\mathbb{R}^{N \times 6}$ cuyos renglones son $\Phi(x) \in \mathbb{R}^6$. ¿Ves alguna relación de esto con la distancia de Mahalanobis?*

**Respuesta:**

¡Sí! Existe una conexión matemática fundamental: el Kernel de Christoffel-Darboux es una **generalización no lineal de la distancia de Mahalanobis al espacio de momentos polinomiales (un espacio de Hilbert con núcleo reproductor o RKHS polinomial)**.

1. **La distancia de Mahalanobis como caso de momentos de grado 1:**  
   La distancia de Mahalanobis al origen se define como:
   $$d_M(x)^2 = x^T \Sigma^{-1} x$$
   Si tomamos la base lineal $x = (x_1, x_2)^T \in \mathbb{R}^2$, la matriz de momentos de primer orden es exactamente la matriz de covarianza muestral $\Sigma = \frac{1}{N} X^T X$.

2. **Extensión a monomios de grado superior:**  
   En el método de Christoffel-Darboux, ampliamos la representación mediante el vector de monomios $\Phi(x) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)^T \in \mathbb{R}^6$. La matriz de momentos empírica es:
   $$M_n = \frac{1}{N} D^T D = \frac{1}{N} \sum_{i=1}^N \Phi(x_i) \Phi(x_i)^T$$
   El polinomio evaluado es:
   $$p_{CD}(x) = \Phi(x)^T M_n^{-1} \Phi(x)$$
   Esta expresión posee **idéntica estructura cuadrática ponderada por la inversa de una matriz de momentos**. Mientras Mahalanobis aproxima los datos mediante hiper-elipsoides gaussianos (momentos de 2º orden), Christoffel-Darboux utiliza momentos empíricos de orden hasta $2d$, construyendo funciones de densidad empírica inversa no lineales capaces de amoldarse a geometrías complejas y no gaussianas.

---

### Pregunta 7: Propiedades Geométricas del Polinomio de Christoffel-Darboux
> *¿El polinomio de Christoffel-Darboux aplicado determina una distancia al origen y los subconjuntos de nivel son bolas con centro en el origen en $\mathbb{R}^2$?*

**Respuesta:**

**No.**

1. **No determina una distancia en $\mathbb{R}^2$:**  
   Para ser una distancia métrica, $p_{CD}(x)$ debería satisfacer los tres axiomas fundamentales de una métrica:
   - *Identidad de indiscernibles:* Debería cumplirse $p_{CD}(0) = 0$. Sin embargo, en el origen $x = (0, 0)$, el vector polinomial es $\Phi(0) = (1, 0, 0, 0, 0, 0)^T$. Por ende:
     $$p_{CD}(0) = \Phi(0)^T M_{\text{reg}}^{-1} \Phi(0) = [M_{\text{reg}}^{-1}]_{11} > 0$$
   - *Homogeneidad:* No cumple $p_{CD}(\alpha x) = |\alpha| p_{CD}(x)$, ya que involucra términos constantes y de orden par ($x_i^2, x_i^4$).
   - *Desigualdad triangular:* No se cumple en $\mathbb{R}^2$.  
   $p_{CD}(x)$ es en realidad un estimador de **densidad inversa espectral** que crece conforme $x$ se aleja del soporte empírico de las transacciones legítimas.

2. **Los subconjuntos de nivel NO son bolas con centro en el origen:**  
   Los subconjuntos $S_\rho = \{x \in \mathbb{R}^2 : p_{CD}(x) \le \rho\}$ son **conjuntos de nivel semialgebraicos** delimitados por curvas polinomiales de grado $2d$ (grado 4 para base de grado 2, y grado 8 para base de grado 4). Estas fronteras:
   - No tienen centro geométrico en $(0, 0)$.
   - Presentan deformaciones, concavidades y asimetrías que se adaptan a la dispersión empírica real de la nube de datos, a diferencia de las elipses canónicas concéntricas de Mahalanobis.

---

### Pregunta 8: Invertibilidad Estricta de la Matriz Regularizada de Tikhonov
> *Para asegurar la invertibilidad de $M$ se suele aplicar la regularización de Tikhonov (ridge): $M_{\text{reg}} = M_n + \epsilon I$ (donde $\epsilon = 10^{-6}$ e $I$ es la matriz identidad). Hacer los cálculos con $M_{\text{reg}}^{-1}$ en lugar de $M^{-1}$. ¿Por qué esta matriz siempre es invertible?*

**Respuesta y Demostración Matemática Rigurosa:**

1. **Semidefinición positiva de la matriz de momentos $M_n$:**  
   Por definición, $M_n = \frac{1}{N} D^T D$, donde $D \in \mathbb{R}^{N \times k}$.  
   Para cualquier vector no nulo $v \in \mathbb{R}^k$ ($v \ne \mathbf{0}$):
   $$v^T M_n v = v^T \left(\frac{1}{N} D^T D\right) v = \frac{1}{N} (Dv)^T (Dv) = \frac{1}{N} \|Dv\|_2^2 \ge 0$$
   Como la norma euclidiana al cuadrado es no negativa, $M_n$ es simétrica y **semidefinida positiva**. Por el Teorema Espectral, todos sus autovalores son reales y no negativos:
   $$\lambda_i(M_n) \ge 0, \quad \forall i \in \{1, 2, \dots, k\}$$

2. **Efecto de la perturbación de Tikhonov ($\epsilon I$):**  
   Sea $M_{\text{reg}} = M_n + \epsilon I$ con $\epsilon = 10^{-6} > 0$. Si $v_i$ es un autovector ortonormal de $M_n$ con autovalor $\lambda_i(M_n)$:
   $$M_{\text{reg}} v_i = (M_n + \epsilon I) v_i = M_n v_i + \epsilon v_i = \lambda_i(M_n) v_i + \epsilon v_i = (\lambda_i(M_n) + \epsilon) v_i$$
   Por tanto, los autovalores de $M_{\text{reg}}$ son:
   $$\lambda_i(M_{\text{reg}}) = \lambda_i(M_n) + \epsilon$$

3. **Cota inferior estricta y rango completo:**  
   Dado que $\lambda_i(M_n) \ge 0$ y $\epsilon > 0$:
   $$\lambda_i(M_{\text{reg}}) \ge \epsilon = 10^{-6} > 0, \quad \forall i \in \{1, \dots, k\}$$
   El determinante de $M_{\text{reg}}$ es el producto de sus autovalores:
   $$\det(M_{\text{reg}}) = \prod_{i=1}^k \lambda_i(M_{\text{reg}}) \ge \epsilon^k > 0$$
   Al ser estrictamente positiva definida, $\ker(M_{\text{reg}}) = \{\mathbf{0}\}$, por lo que $M_{\text{reg}}$ tiene rango completo ($k$). Por el **Teorema de la Matriz Invertible**, $M_{\text{reg}}$ es **siempre invertible**, su inversa $M_{\text{reg}}^{-1}$ existe, es única y está numéricamente estabilizada ante colinealidades en la base de momentos.
