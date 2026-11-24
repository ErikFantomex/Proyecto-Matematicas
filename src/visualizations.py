"""
Módulo de visualización para el Proyecto Final de Matemáticas para Ciencia de Datos.
Genera las gráficas en dos dimensiones requeridas en el documento:
1. Elipses de Mahalanobis (2D)
2. Fronteras del Kernel CD Grado 2 (2D)
3. Fronteras del Kernel CD Grado 4 (2D) - Réplica de la portada del PDF
4. Panel comparativo de métricas, ROC y varianza explicada.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


def setup_style():
    """Configura el estilo estético de las gráficas."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8


def plot_mahalanobis_2d(detector, X_legit, X_fraud, output_path="plots/mahalanobis_2d.png"):
    """
    Grafica las elipses de Mahalanobis para los radios rho_50, rho_75, rho_90, rho_95, rho_99.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Determinar límites
    all_x = np.vstack([X_legit[:, :2], X_fraud[:, :2]])
    x_min, x_max = all_x[:, 0].min() - 1.5, all_x[:, 0].max() + 1.5
    y_min, y_max = all_x[:, 1].min() - 1.5, all_x[:, 1].max() + 1.5
    
    # Malla para evaluar la distancia
    grid_res = 300
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    d_grid = detector.compute_distance(grid_pts).reshape(xx.shape)
    
    # Dibujar puntos
    ax.scatter(X_legit[:, 0], X_legit[:, 1], c='#1f77b4', alpha=0.45, s=20, 
               edgecolors='none', label='Transacciones Legítimas')
    ax.scatter(X_fraud[:, 0], X_fraud[:, 1], c='#d62728', alpha=0.9, s=35, 
               edgecolors='black', linewidth=0.5, label='Fraude')
    
    # Dibujar elipses de nivel
    percentiles = [50, 75, 90, 95, 99]
    colors = ['#2ca02c', '#17becf', '#ff7f0e', '#9467bd', '#8c564b']
    linestyles = [':', '-.', '--', '-', '-']
    
    radii_vals = [detector.find_radius(p) for p in percentiles]
    
    cs = ax.contour(xx, yy, d_grid, levels=radii_vals, colors=colors, linestyles=linestyles, linewidths=1.8)
    
    # Etiquetas de niveles
    fmt = {}
    for r, p in zip(radii_vals, percentiles):
        fmt[r] = f'{p}% ({r:.2f})'
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=9)
    
    ax.set_title("Fronteras de Decisión - Distancia de Mahalanobis (2D)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("PC1 (Estandarizada)", fontsize=12)
    ax.set_ylabel("PC2 (Estandarizada)", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfica guardada: {output_path}")


def plot_cd_kernel_2d(detector, X_legit, X_fraud, degree=4, levels=[90, 95, 99], 
                      output_path="plots/cd_kernel_deg4_2d.png"):
    """
    Grafica los subconjuntos de nivel del Kernel de Christoffel-Darboux.
    Réplica exacta del formato de la portada del PDF.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Límites específicos centrados en la nube principal
    x_min, x_max = -2.5, 14.5
    y_min, y_max = -3.5, 7.5
    
    grid_res = 350
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    p_grid = detector.compute_pcd(grid_pts).reshape(xx.shape)
    
    # Puntos legítimos y fraude
    ax.scatter(X_legit[:, 0], X_legit[:, 1], c='#3b82f6', alpha=0.35, s=16, 
               edgecolors='none', label='Transacciones Legítimas')
    ax.scatter(X_fraud[:, 0], X_fraud[:, 1], c='#b91c1c', alpha=0.9, s=32, 
               edgecolors='black', linewidth=0.4, label='Fraude')
    
    # Niveles solicitados
    radii_vals = [detector.find_radius(p) for p in levels]
    contour_colors = ['#0ea5e9', '#6366f1', '#4338ca']
    
    cs = ax.contour(xx, yy, p_grid, levels=radii_vals, colors=contour_colors, linewidths=1.8)
    
    fmt = {}
    for r, p in zip(radii_vals, levels):
        fmt[r] = f'{r:.3f}'
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=8)
    
    # Leyenda personalizada con las fronteras
    lines = [
        plt.Line2D([0], [0], color='#0ea5e9', lw=1.8),
        plt.Line2D([0], [0], color='#6366f1', lw=1.8),
        plt.Line2D([0], [0], color='#4338ca', lw=1.8),
    ]
    labels = [f'Frontera Densidad {p}%' for p in levels]
    
    handles, orig_labels = ax.get_legend_handles_labels()
    ax.legend(handles + lines, orig_labels + labels, loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    
    ax.set_title(f"Fronteras de Decisión (3 Niveles del Kernel CD) - Grado {degree}", 
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("PC1 (Estandarizada)", fontsize=11)
    ax.set_ylabel("PC2 (Estandarizada)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfica guardada: {output_path}")


def plot_comparative_dashboard(benchmark_results, pca_variance_ratio, output_path="plots/comparative_dashboard.png"):
    """
    Genera un panel comparativo de 4 subgráficas:
    1. Varianza Explicada Acumulada de PCA (10 componentes).
    2. Curvas ROC de todos los métodos probados en X_test.
    3. Tiempo de ejecución (ms) por método.
    4. Balance Precisión, Recall y F1-score.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    # 1. Varianza explicada
    ax1 = axes[0, 0]
    comps = np.arange(1, 11)
    cum_var = np.cumsum(pca_variance_ratio) * 100
    indiv_var = pca_variance_ratio * 100
    
    ax1.bar(comps, indiv_var, alpha=0.6, color='#2563eb', label='Varianza Individual')
    ax1.plot(comps, cum_var, marker='o', color='#dc2626', linewidth=2, label='Varianza Acumulada')
    ax1.axhline(cum_var[1], color='gray', linestyle=':', label=f'2 Comp ({cum_var[1]:.1f}%)')
    ax1.axhline(cum_var[9], color='black', linestyle='--', label=f'10 Comp ({cum_var[9]:.1f}%)')
    ax1.set_title("Varianza Explicada por Componente Principal", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Número de Componentes")
    ax1.set_ylabel("Porcentaje de Varianza (%)")
    ax1.set_xticks(comps)
    ax1.legend(loc='center right', fontsize=8)
    
    # 2. Curvas ROC
    ax2 = axes[0, 1]
    from sklearn.metrics import roc_curve
    for res in benchmark_results:
        fpr, tpr, _ = roc_curve(res['y_test'], res['scores'])
        ax2.plot(fpr, tpr, lw=1.8, label=f"{res['name']} (AUC = {res['roc_auc']:.2f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    ax2.set_title("Curvas ROC en Conjunto de Prueba (X_test_pca)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax2.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax2.legend(loc='lower right', fontsize=8)
    
    # 3. Tiempos de ejecución
    ax3 = axes[1, 0]
    names = [r['name'] for r in benchmark_results]
    times = [r['exec_time_ms'] for r in benchmark_results]
    y_pos = np.arange(len(names))
    colors = ['#3b82f6' if 'Maha' in n else '#10b981' for n in names]
    
    bars = ax3.barh(y_pos, times, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel("Tiempo de Ejecución (ms)")
    ax3.set_title("Tiempo de Inferencia en Prueba (ms)", fontsize=12, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax3.text(w + 0.1, bar.get_y() + bar.get_height()/2, f'{w:.2f}ms', va='center', fontsize=8)
        
    # 4. Métricas de Clasificación (F1, Precision, Recall)
    ax4 = axes[1, 1]
    x = np.arange(len(names))
    width = 0.25
    
    f1s = [r['f1_score'] for r in benchmark_results]
    recs = [r['recall'] for r in benchmark_results]
    precs = [r['precision'] for r in benchmark_results]
    
    ax4.bar(x - width, precs, width, label='Precisión', color='#6366f1', alpha=0.85)
    ax4.bar(x, recs, width, label='Recall', color='#f59e0b', alpha=0.85)
    ax4.bar(x + width, f1s, width, label='F1-Score', color='#ec4899', alpha=0.85)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.replace(' (Deg 2)', '').replace(' (Deg 4)', '-D4') for n in names], 
                        rotation=30, ha='right', fontsize=8)
    ax4.set_ylabel("Métrica (0 a 1)")
    ax4.set_title("Comparativa de Precisión, Recall y F1 (Umbral 95%)", fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Dashboard comparativo guardado: {output_path}")
