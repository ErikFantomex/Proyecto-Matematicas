"""
Script principal del Proyecto Final: Matemáticas para Ciencia de Datos.
Detección de Fraude Financiero mediante Distancia de Mahalanobis y Kernel de Christoffel-Darboux.
"""

import os
import json
import numpy as np
import pandas as pd

from src.data_loader import load_and_preprocess_data
from src.mahalanobis import MahalanobisDetector
from src.christoffel_darboux import ChristoffelDarbouxDetector
from src.visualizations import (
    plot_mahalanobis_2d, 
    plot_cd_kernel_2d, 
    plot_comparative_dashboard
)


def main():
    print("=" * 80)
    print("PROYECTO FINAL: MATEMÁTICAS PARA CIENCIA DE DATOS")
    print("DETECCIÓN DE FRAUDE MEDIANTE MAHALANOBIS Y KERNEL CHRISTOFFEL-DARBOUX")
    print("=" * 80)
    
    # 1. Cargar y preparar datos (Sección 1 del PDF)
    print("\n[Paso 1] Cargando y preparando dataset...")
    data = load_and_preprocess_data(test_size=0.2, random_state=42)
    
    X_train_pca = data['X_train_pca']
    y_train = data['y_train']
    X_test_pca = data['X_test_pca']
    y_test = data['y_test']
    X_train_legit = data['X_train_legit']
    explained_var = data['explained_variance_ratio']
    cum_var = data['cumulative_variance']
    
    X_train_fraud = X_train_pca[y_train == 1]
    
    print(f"-> Muestras Entrenamiento: {len(X_train_pca)} (Legítimas: {len(X_train_legit)}, Fraude: {len(X_train_fraud)})")
    print(f"-> Muestras Prueba: {len(X_test_pca)} (Legítimas: {(y_test == 0).sum()}, Fraude: {(y_test == 1).sum()})")
    print(f"-> Varianza explicada por 2 componentes: {cum_var[1]*100:.2f}%")
    print(f"-> Varianza explicada por 5 componentes: {cum_var[4]*100:.2f}%")
    print(f"-> Varianza explicada por 7 componentes: {cum_var[6]*100:.2f}%")
    print(f"-> Varianza explicada por 10 componentes: {cum_var[9]*100:.2f}%")

    benchmark_list = []
    
    # 2. Primera propuesta: Mahalanobis (Sección 2 del PDF)
    print("\n" + "=" * 50)
    print("[Sección 2] PRIMERA PROPUESTA: DISTANCIA DE MAHALANOBIS")
    print("=" * 50)
    
    # Mahalanobis 2D
    maha_2d = MahalanobisDetector(n_components=2)
    maha_2d.fit(X_train_legit)
    
    print("\n--- Mahalanobis (2 Componentes) ---")
    percentiles = [50, 75, 90, 95, 99]
    print("Radios rho_n encontrados sobre X_train_legit:")
    for p in percentiles:
        r = maha_2d.find_radius(p)
        print(f"  rho_{p} (radio para {p}% legítimos): {r:.4f}")
        
    zones_2d = maha_2d.compute_risk_zones(X_train_fraud)
    print("\nDistribución de transacciones fraudulentas de entrenamiento en zonas de riesgo (2D):")
    print(f"  - Seguridad Extrema (d <= rho_50):      {zones_2d['extrema']} ({zones_2d['extrema']/zones_2d['total_frauds']*100:.1f}%)")
    print(f"  - Seguridad Alta    (rho_50 < d <= rho_75): {zones_2d['alta']} ({zones_2d['alta']/zones_2d['total_frauds']*100:.1f}%)")
    print(f"  - Seguridad Baja    (rho_75 < d <= rho_90): {zones_2d['baja']} ({zones_2d['baja']/zones_2d['total_frauds']*100:.1f}%)")
    print(f"  - Zona de Peligro   (d > rho_90):          {zones_2d['peligro']} ({zones_2d['peligro']/zones_2d['total_frauds']*100:.1f}%)")
    print(f"  - Peligro Extremo   (d > rho_95):          {zones_2d['peligro_extremo']} ({zones_2d['peligro_extremo']/zones_2d['total_frauds']*100:.1f}%)")
    
    eval_m2 = maha_2d.evaluate(X_test_pca, y_test, threshold_percentile=95)
    print(f"\nEvaluación en X_test_pca (Umbral 95%):")
    print(f"  Tiempo ejecución: {eval_m2['exec_time_ms']:.2f} ms")
    print(f"  Matriz de Confusión (en un renglón): {eval_m2['cm_line']}")
    print(f"  Precisión: {eval_m2['precision']:.4f} | Recall: {eval_m2['recall']:.4f} | F1: {eval_m2['f1_score']:.4f} | ROC-AUC: {eval_m2['roc_auc']:.4f}")
    
    benchmark_list.append({
        'name': 'Mahalanobis 2D',
        'n_components': 2,
        'degree': 1,
        'var_exp': cum_var[1] * 100,
        'exec_time_ms': eval_m2['exec_time_ms'],
        'cm_line': eval_m2['cm_line'],
        'precision': eval_m2['precision'],
        'recall': eval_m2['recall'],
        'f1_score': eval_m2['f1_score'],
        'roc_auc': eval_m2['roc_auc'],
        'scores': eval_m2['distances'],
        'y_test': y_test
    })

    # Mahalanobis 10D
    maha_10d = MahalanobisDetector(n_components=10)
    maha_10d.fit(X_train_legit)
    eval_m10 = maha_10d.evaluate(X_test_pca, y_test, threshold_percentile=95)
    print("\n--- Mahalanobis (10 Componentes) ---")
    for p in percentiles:
        r = maha_10d.find_radius(p)
        print(f"  rho_{p}: {r:.4f}")
    print(f"Matriz de Confusión (un renglón): {eval_m10['cm_line']}")
    print(f"Precisión: {eval_m10['precision']:.4f} | Recall: {eval_m10['recall']:.4f} | F1: {eval_m10['f1_score']:.4f} | ROC-AUC: {eval_m10['roc_auc']:.4f}")
    
    benchmark_list.append({
        'name': 'Mahalanobis 10D',
        'n_components': 10,
        'degree': 1,
        'var_exp': cum_var[9] * 100,
        'exec_time_ms': eval_m10['exec_time_ms'],
        'cm_line': eval_m10['cm_line'],
        'precision': eval_m10['precision'],
        'recall': eval_m10['recall'],
        'f1_score': eval_m10['f1_score'],
        'roc_auc': eval_m10['roc_auc'],
        'scores': eval_m10['distances'],
        'y_test': y_test
    })

    # 3. Segunda propuesta: Kernel de Christoffel-Darboux (Sección 3 del PDF)
    print("\n" + "=" * 50)
    print("[Sección 3] SEGUNDA PROPUESTA: KERNEL DE CHRISTOFFEL-DARBOUX")
    print("=" * 50)
    
    cd_configs = [
        ("CD 2D (Deg 2)", 2, 2, cum_var[1]),
        ("CD 2D (Deg 4)", 2, 4, cum_var[1]),
        ("CD 5D (Deg 2)", 5, 2, cum_var[4]),
        ("CD 7D (Deg 2)", 7, 2, cum_var[6]),
        ("CD 10D (Deg 2)", 10, 2, cum_var[9]),
    ]
    
    cd_detectors = {}
    
    for name, dims, deg, var_val in cd_configs:
        print(f"\n--- {name} (Dim={dims}, Grado={deg}) ---")
        cd = ChristoffelDarbouxDetector(n_components=dims, degree=deg, epsilon=1e-6)
        cd.fit(X_train_legit)
        cd_detectors[name] = cd
        
        print(f"Dimensión de la base polinomial (k): {cd.feature_dim}")
        print("Radios rho_n (subconjuntos de nivel):")
        for p in percentiles:
            r = cd.find_radius(p)
            print(f"  rho_{p}: {r:.4f}")
            
        eval_cd = cd.evaluate(X_test_pca, y_test, threshold_percentile=95)
        print(f"Matriz de Confusión (un renglón): {eval_cd['cm_line']}")
        print(f"Tiempo: {eval_cd['exec_time_ms']:.2f} ms | Prec: {eval_cd['precision']:.4f} | Rec: {eval_cd['recall']:.4f} | F1: {eval_cd['f1_score']:.4f} | AUC: {eval_cd['roc_auc']:.4f}")
        
        benchmark_list.append({
            'name': name,
            'n_components': dims,
            'degree': deg,
            'var_exp': var_val * 100,
            'exec_time_ms': eval_cd['exec_time_ms'],
            'cm_line': eval_cd['cm_line'],
            'precision': eval_cd['precision'],
            'recall': eval_cd['recall'],
            'f1_score': eval_cd['f1_score'],
            'roc_auc': eval_cd['roc_auc'],
            'scores': eval_cd['scores'],
            'y_test': y_test
        })

    # 4. Generar y guardar gráficas
    print("\n" + "=" * 50)
    print("[Paso 4] GENERANDO GRÁFICAS EN DOS DIMENSIONES")
    print("=" * 50)
    
    # 4.1 Mahalanobis 2D
    plot_mahalanobis_2d(maha_2d, X_train_legit, X_train_fraud, 
                        output_path="plots/mahalanobis_2d_ellipses.png")
    
    # 4.2 CD 2D Grado 2
    plot_cd_kernel_2d(cd_detectors["CD 2D (Deg 2)"], X_train_legit, X_train_fraud, 
                      degree=2, levels=[90, 95, 99],
                      output_path="plots/christoffel_darboux_deg2_2d.png")
    
    # 4.3 CD 2D Grado 4 (Portada del PDF)
    plot_cd_kernel_2d(cd_detectors["CD 2D (Deg 4)"], X_train_legit, X_train_fraud, 
                      degree=4, levels=[90, 95, 99],
                      output_path="plots/christoffel_darboux_deg4_2d.png")
    
    # 4.4 Dashboard comparativo
    plot_comparative_dashboard(benchmark_list, explained_var, 
                              output_path="plots/comparative_performance.png")

    # 5. Tabla comparativa final (Entregable 1)
    print("\n" + "=" * 90)
    print("TABLA COMPARATIVA FINAL (X_test_pca, Umbral 95%):")
    print("=" * 90)
    header = f"{'Método':<16} | {'Dim':<4} | {'VarExp(%)':<9} | {'Tiempo(ms)':<10} | {'Matriz Confusión (1 renglón)':<26} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6}"
    print(header)
    print("-" * len(header))
    for r in benchmark_list:
        row = f"{r['name']:<16} | {r['n_components']:<4} | {r['var_exp']:<9.1f} | {r['exec_time_ms']:<10.2f} | {r['cm_line']:<26} | {r['precision']:<6.3f} | {r['recall']:<6.3f} | {r['f1_score']:<6.3f} | {r['roc_auc']:<6.3f}"
        print(row)
    print("=" * 90)

    # Exportar resultados a JSON para el reporte
    clean_bench = []
    for r in benchmark_list:
        d = dict(r)
        del d['scores']
        del d['y_test']
        clean_bench.append(d)
        
    with open("output/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_bench, f, indent=2, ensure_ascii=False)
        
    print("\nProceso completado exitosamente. Todos los entregables generados.")


if __name__ == "__main__":
    main()
