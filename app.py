"""
Tablero Interactivo en Streamlit: Detección de Fraude Financiero
Matemáticas para Ciencia de Datos — Octubre 2026
Distancia de Mahalanobis y Kernel de Christoffel-Darboux
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from src.data_loader import load_and_preprocess_data
from src.mahalanobis import MahalanobisDetector
from src.christoffel_darboux import ChristoffelDarbouxDetector

# Configuración de página
st.set_page_config(
    page_title="Detección de Fraude - Kernel Christoffel-Darboux & Mahalanobis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_data():
    """Carga y procesa los datos una sola vez."""
    return load_and_preprocess_data(test_size=0.2, random_state=42)


@st.cache_resource
def get_trained_models(_data):
    """Entrena los detectores y los mantiene en caché."""
    X_train_legit = _data['X_train_legit']
    
    models = {
        'Mahalanobis 2D': MahalanobisDetector(n_components=2).fit(X_train_legit),
        'Mahalanobis 10D': MahalanobisDetector(n_components=10).fit(X_train_legit),
        'CD 2D (Grado 2)': ChristoffelDarbouxDetector(n_components=2, degree=2, epsilon=1e-6).fit(X_train_legit),
        'CD 2D (Grado 4)': ChristoffelDarbouxDetector(n_components=2, degree=4, epsilon=1e-6).fit(X_train_legit),
        'CD 5D (Grado 2)': ChristoffelDarbouxDetector(n_components=5, degree=2, epsilon=1e-6).fit(X_train_legit),
        'CD 7D (Grado 2)': ChristoffelDarbouxDetector(n_components=7, degree=2, epsilon=1e-6).fit(X_train_legit),
        'CD 10D (Grado 2)': ChristoffelDarbouxDetector(n_components=10, degree=2, epsilon=1e-6).fit(X_train_legit),
    }
    return models


# Carga de datos y modelos
with st.spinner("Inicializando modelos matemáticos y cargando dataset..."):
    data = get_cached_data()
    models = get_trained_models(data)

# Variables principales
X_train_pca = data['X_train_pca']
y_train = data['y_train']
X_test_pca = data['X_test_pca']
y_test = data['y_test']
X_train_legit = data['X_train_legit']
X_train_fraud = X_train_pca[y_train == 1]
cum_var = data['cumulative_variance'] * 100
exp_var = data['explained_variance_ratio'] * 100

# Header
st.markdown('<div class="main-header">🛡️ Detección de Fraude Financiero</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modelado de Transacciones con Distancia de Mahalanobis y Kernel de Christoffel-Darboux — Departamento de Matemáticas</div>', unsafe_allow_html=True)

# Barra lateral con controles globales
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Configuración Global")
global_threshold = st.sidebar.slider(
    "Umbral de Decisión (% de densidad legítima):",
    min_value=50, max_value=99, value=95, step=1,
    help="Determina el radio rho_n en el que se clasifica como anomalía/fraude si la distancia o kernel excede el percentil n% de las transacciones legítimas."
)
st.sidebar.info(f"**Umbral activo:** $\\rho_{{{global_threshold}}}$\n\n- Dentro de la frontera $\\le \\rho_{{{global_threshold}}}$: **Legítima**\n- Fuera de la frontera $> \\rho_{{{global_threshold}}}$: **Fraude (Alerta)**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Resumen del Dataset")
st.sidebar.write(f"- **País filtrado:** México")
st.sidebar.write(f"- **Total transacciones:** {len(data['df_mexico']):,}")
st.sidebar.write(f"- **Entrenamiento:** {len(X_train_pca):,} (Fraudes: {y_train.sum()})")
st.sidebar.write(f"- **Prueba:** {len(X_test_pca):,} (Fraudes: {y_test.sum()})")
st.sidebar.write(f"- **Variables numéricas:** {len(data['feature_cols'])}")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualizador 2D de Fronteras",
    "📈 Comparativa Global & Métricas",
    "🔍 Simulador de Transacciones",
    "📚 Fundamentos Matemáticos"
])

# -------------------------------------------------------------
# TAB 1: Visualizador 2D de Fronteras
# -------------------------------------------------------------
with tab1:
    st.subheader("Visualización Bidimensional de Fronteras de Decisión")
    st.write("Explora la geometría que delimita las transacciones legítimas frente a los fraudes en las dos componentes principales dominantes.")
    
    col_sel, col_stat = st.columns([2, 1])
    with col_sel:
        selected_model_name = st.selectbox(
            "Selecciona el Modelo a Visualizar en 2D:",
            ["CD 2D (Grado 4)", "CD 2D (Grado 2)", "Mahalanobis 2D"],
            index=0
        )
    with col_stat:
        st.metric("Varianza Explicada (2 Comp.)", f"{cum_var[1]:.2f}%", delta="21.17% del total", delta_color="off")

    active_model = models[selected_model_name]
    
    # Calcular radios para niveles 90, 95, 99 y el umbral global
    levels_to_plot = sorted(list(set([90, 95, 99, global_threshold])))
    radii_to_plot = [active_model.find_radius(p) for p in levels_to_plot]
    
    # Malla para contornos
    x_min, x_max = -2.5, 14.5
    y_min, y_max = -3.5, 7.5
    res = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    if "Mahalanobis" in selected_model_name:
        z_grid = active_model.compute_distance(grid_pts).reshape(xx.shape)
    else:
        z_grid = active_model.compute_pcd(grid_pts).reshape(xx.shape)

    # Gráfico con Matplotlib integrado
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    
    # Puntos legítimos y fraudes
    ax.scatter(X_train_legit[:, 0], X_train_legit[:, 1], c='#3b82f6', alpha=0.35, s=18, label='Legítimas (Train)')
    ax.scatter(X_train_fraud[:, 0], X_train_fraud[:, 1], c='#dc2626', alpha=0.9, s=35, edgecolors='black', label='Fraude (Train)')
    
    # Contornos de nivel
    cmap_colors = ['#0ea5e9', '#6366f1', '#4338ca', '#f59e0b']
    colors_chosen = cmap_colors[:len(levels_to_plot)]
    cs = ax.contour(xx, yy, z_grid, levels=radii_to_plot, colors=colors_chosen, linewidths=2.0)
    
    fmt = {r: f"{p}% ({r:.2f})" for r, p in zip(radii_to_plot, levels_to_plot)}
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=8)
    
    ax.set_title(f"Fronteras de Decisión en 2D — {selected_model_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC1 (Estandarizada)")
    ax.set_ylabel("PC2 (Estandarizada)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("""
    > [!TIP] **Observación geométrica clave:**  
    > Mientras que **Mahalanobis 2D** genera elipses concéntricas forzando una simetría gaussiana, el **Kernel de Christoffel-Darboux Grado 4** genera curvas de nivel semialgebraicas que se flexionan adaptándose a la verdadera forma asimétrica y no convexa del conjunto de datos.
    """)

# -------------------------------------------------------------
# TAB 2: Comparativa Global & Métricas
# -------------------------------------------------------------
with tab2:
    st.subheader(f"Tabla Comparativa de Modelos en Conjunto de Prueba (`X_test_pca`, Umbral {global_threshold}%)")
    
    benchmark_data = []
    roc_curves_data = []
    
    for name, detector in models.items():
        dims = detector.n_components
        var_e = cum_var[dims - 1]
        
        # Evaluar
        eval_res = detector.evaluate(X_test_pca, y_test, threshold_percentile=global_threshold)
        scores = eval_res['distances'] if "Mahalanobis" in name else eval_res['scores']
        
        benchmark_data.append({
            "Método": name,
            "Dimensión": dims,
            "Varianza Explicada (%)": f"{var_e:.2f}%",
            "Tiempo Inferencia": f"{eval_res['exec_time_ms']:.2f} ms",
            "Matriz Confusión (1 renglón)": eval_res['cm_line'],
            "Precisión": round(eval_res['precision'], 4),
            "Recall": round(eval_res['recall'], 4),
            "F1-Score": round(eval_res['f1_score'], 4),
            "ROC-AUC": round(eval_res['roc_auc'], 4),
            "raw_time": eval_res['exec_time_ms'],
            "raw_f1": eval_res['f1_score'],
            "raw_rec": eval_res['recall'],
            "raw_auc": eval_res['roc_auc']
        })
        
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_curves_data.append((name, fpr, tpr, eval_res['roc_auc']))

    df_bench = pd.DataFrame(benchmark_data)
    
    # Mostrar tabla estilizada
    st.dataframe(
        df_bench[["Método", "Dimensión", "Varianza Explicada (%)", "Tiempo Inferencia", 
                  "Matriz Confusión (1 renglón)", "Precisión", "Recall", "F1-Score", "ROC-AUC"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Gráficas interactivas con Plotly
    col_roc, col_bars = st.columns(2)
    
    with col_roc:
        st.markdown("#### Curvas ROC en Conjunto de Prueba")
        fig_roc = go.Figure()
        for name, fpr, tpr, auc_val in roc_curves_data:
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc_val:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name="Aleatorio"))
        fig_roc.update_layout(
            xaxis_title="Tasa de Falsos Positivos (FPR)",
            yaxis_title="Tasa de Verdaderos Positivos (TPR)",
            legend=dict(x=0.45, y=0.05, font=dict(size=9)),
            margin=dict(l=40, r=20, t=30, b=40),
            height=380
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_bars:
        st.markdown("#### Comparativa de Recall y ROC-AUC")
        fig_bar = go.Figure()
        names = [d["Método"] for d in benchmark_data]
        recs = [d["raw_rec"] for d in benchmark_data]
        aucs = [d["raw_auc"] for d in benchmark_data]
        
        fig_bar.add_trace(go.Bar(x=names, y=recs, name="Recall", marker_color="#f59e0b"))
        fig_bar.add_trace(go.Bar(x=names, y=aucs, name="ROC-AUC", marker_color="#3b82f6"))
        fig_bar.update_layout(
            barmode='group',
            xaxis_tickangle=-30,
            yaxis=dict(range=[0, 1.05], title="Métrica (0 a 1)"),
            legend=dict(x=0.05, y=0.95),
            margin=dict(l=40, r=20, t=30, b=80),
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Simulador de Transacciones en Vivo
# -------------------------------------------------------------
with tab3:
    st.subheader("Simulador y Auditoría de Transacciones Individuales")
    st.write("Selecciona una transacción del conjunto de prueba o simula valores para evaluar su nivel de seguridad en tiempo real.")
    
    sample_idx = st.slider("Índice de muestra de prueba (`X_test`):", 0, len(X_test_pca)-1, 12)
    sample_vec = X_test_pca[sample_idx:sample_idx+1]
    is_actual_fraud = y_test[sample_idx] == 1
    
    st.markdown(f"**Etiqueta Real:** {'🚨 FRAUDE' if is_actual_fraud else '✅ TRANSACCIÓN LEGÍTIMA'}")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    # Evaluar con Mahalanobis 10D
    m10 = models['Mahalanobis 10D']
    d_val = m10.compute_distance(sample_vec)[0]
    r95_m = m10.find_radius(global_threshold)
    is_alert_m = d_val > r95_m
    with col_m1:
        st.markdown("### Mahalanobis 10D")
        st.metric("Distancia $d_M(x)$", f"{d_val:.3f}", delta=f"Umbral: {r95_m:.3f}", delta_color="inverse")
        if is_alert_m:
            st.error("🚨 ALERTA: Fraude Detectado")
        else:
            st.success("🟢 Transacción Aceptada")
            
    # Evaluar con CD 2D Grado 4
    cd4 = models['CD 2D (Grado 4)']
    p4_val = cd4.compute_pcd(sample_vec)[0]
    r95_cd4 = cd4.find_radius(global_threshold)
    is_alert_cd4 = p4_val > r95_cd4
    with col_m2:
        st.markdown("### CD 2D (Grado 4)")
        st.metric("Polinomio $p_{CD}(x)$", f"{p4_val:.3f}", delta=f"Umbral: {r95_cd4:.3f}", delta_color="inverse")
        if is_alert_cd4:
            st.error("🚨 ALERTA: Fraude Detectado")
        else:
            st.success("🟢 Transacción Aceptada")

    # Evaluar con CD 10D Grado 2
    cd10 = models['CD 10D (Grado 2)']
    p10_val = cd10.compute_pcd(sample_vec)[0]
    r95_cd10 = cd10.find_radius(global_threshold)
    is_alert_cd10 = p10_val > r95_cd10
    with col_m3:
        st.markdown("### CD 10D (Grado 2)")
        st.metric("Polinomio $p_{CD}(x)$", f"{p10_val:.3f}", delta=f"Umbral: {r95_cd10:.3f}", delta_color="inverse")
        if is_alert_cd10:
            st.error("🚨 ALERTA: Fraude Detectado")
        else:
            st.success("🟢 Transacción Aceptada")

# -------------------------------------------------------------
# TAB 4: Fundamentos Matemáticos
# -------------------------------------------------------------
with tab4:
    st.subheader("Fundamentos Matemáticos y Respuestas Teóricas")
    
    with st.expander("1. ¿Por qué es fundamental estandarizar los datos antes de PCA?", expanded=True):
        st.markdown("""
        Las variables de transacciones financieras tienen unidades y magnitudes heterogéneas (`transaction_amount` en miles, `account_age_days` en cientos, variables binarias en $\{0, 1\}$).  
        PCA busca las direcciones ortogonales de máxima varianza. Sin estandarización ($z_j = \\frac{x_j - \\mu_j}{\\sigma_j}$), las columnas con números grandes dominarían artificialmente la covarianza muestral. Estandarizar convierte la matriz de covarianza en la **matriz de correlación**, otorgando a cada variable una ponderación justa.
        """)

    with st.expander("2. Interpretación de las 10 componentes en términos de la SVD"):
        st.markdown("""
        Dada la matriz centrada y estandarizada $X \\in \\mathbb{R}^{N \\times p}$, su SVD es:
        $$X = U \\Sigma V^T$$
        - **Direcciones principales (Loadings):** Son las 10 primeras columnas de $V$, $V_{10} = [v_1, \\dots, v_{10}] \\in \\mathbb{R}^{p \\times 10}$, correspondientes a los 10 valores singulares mayores $\\sigma_1 \\ge \\dots \\ge \\sigma_{10}$.
        - **Coordenadas proyectadas (Scores / $X_{\\text{train\\_pca}}$):**
        $$Z_{10} = X V_{10} = U_{10} \\Sigma_{10} \\in \\mathbb{R}^{N \\times 10}$$
        Representan la proyección óptima en el subespacio de dimensión 10.
        """)

    with st.expander("3. Conexión entre la Distancia de Mahalanobis y la Matriz de Momentos"):
        st.markdown("""
        La distancia de Mahalanobis centrada en el origen es:
        $$d_M(x)^2 = x^T \\Sigma^{-1} x$$
        donde $\\Sigma = \\frac{1}{N} X^T X$ es la **matriz de momentos de orden 1** de las transacciones legítimas.  
        El **Kernel de Christoffel-Darboux** generaliza esto al espacio de monomios polinomiales $\\Phi(x)$:
        $$p_{CD}(x) = \\Phi(x)^T M_n^{-1} \\Phi(x)$$
        donde $M_n = \\frac{1}{N} D^T D$ es la **matriz de momentos de orden $2d$**. Por tanto, el Kernel CD es una generalización semialgebraica y no lineal de la distancia de Mahalanobis.
        """)

    with st.expander("4. Demostración Matemática: ¿Por qué $M_{\\text{reg}} = M_n + \\epsilon I$ siempre es invertible?"):
        st.markdown("""
        1. **Semidefinición positiva:** Para cualquier vector $v \\ne \\mathbf{0}$:
        $$v^T M_n v = v^T \\left(\\frac{1}{N} D^T D\\right) v = \\frac{1}{N} \\|Dv\\|_2^2 \\ge 0$$
        Por tanto, todos los autovalores de $M_n$ satisfacen $\\lambda_i(M_n) \\ge 0$.
        2. **Perturbación de Tikhonov:** Los autovalores de $M_{\\text{reg}} = M_n + \\epsilon I$ son exactamente:
        $$\\lambda_i(M_{\\text{reg}}) = \\lambda_i(M_n) + \\epsilon$$
        3. **Estricta positividad:** Como $\\lambda_i(M_n) \\ge 0$ y $\\epsilon = 10^{-6} > 0$:
        $$\\lambda_i(M_{\\text{reg}}) \\ge \\epsilon > 0, \\quad \\forall i \\in \\{1, \\dots, k\\}$$
        El determinante es $\\det(M_{\\text{reg}}) = \\prod_{i=1}^k \\lambda_i(M_{\\text{reg}}) \\ge \\epsilon^k > 0$. Al no tener autovalores nulos, $M_{\\text{reg}}$ tiene rango completo y **siempre es estrictamente invertible**.
        """)
