"""
Módulo de carga y preprocesamiento de datos para el Proyecto Final de Matemáticas para Ciencia de Datos.
Detección de Fraude Financiero mediante Distancia de Mahalanobis y Kernel de Christoffel-Darboux.
"""

import os
import glob
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_dataset_path():
    """Obtiene la ruta del archivo CSV descargándolo vía kagglehub si es necesario."""
    cache_base = os.path.expanduser(r"~\.cache\kagglehub\datasets\mohankrishnathalla\credit-card-fraud-detection-dataset-2026")
    matches = glob.glob(os.path.join(cache_base, "**", "credit_card_fraud_detection_2026.csv"), recursive=True)
    if matches:
        return matches[0]
    
    # Si no se encuentra, descargar
    path = kagglehub.dataset_download('mohankrishnathalla/credit-card-fraud-detection-dataset-2026')
    csv_path = os.path.join(path, "credit_card_fraud_detection_2026.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    # Buscar en subdirectorios
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.csv'):
                return os.path.join(root, f)
    raise FileNotFoundError("No se encontró el dataset de fraude.")


def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Realiza los pasos 1 a 5 de la Sección 1 del proyecto:
    1. Filtra solo las transacciones de México (merchant_country == 'Mexico').
    2. Considera solo columnas numéricas y la columna is_fraud.
    3. Separa 20% test y 80% train.
    4. Aplica StandardScaler y PCA(n_components=10) ajustados en X_train.
    5. Transforma X_test usando el scaler y PCA aprendidos.
    """
    csv_path = get_dataset_path()
    df = pd.read_csv(csv_path)
    
    # Paso 1: Filtrar México
    df_mexico = df[df['merchant_country'] == 'Mexico'].copy()
    
    # Paso 2: Columnas numéricas e is_fraud
    # Las columnas numéricas predictoras no deben incluir is_fraud
    all_num_cols = df_mexico.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in all_num_cols if c != 'is_fraud']
    
    # Paso 3: Separación 80% train / 20% test
    train_df, test_df = train_test_split(
        df_mexico, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df_mexico['is_fraud']
    )
    
    X_train_raw = train_df[feature_cols].values
    y_train = train_df['is_fraud'].values
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df['is_fraud'].values
    
    # Paso 4: Estandarización y PCA ajustados solo en Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Paso 5: Transformación en Test (usando scaler y pca de entrenamiento)
    X_test_scaled = scaler.transform(X_test_raw)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Filtrar transacciones legítimas de entrenamiento
    legit_mask_train = (y_train == 0)
    X_train_legit = X_train_pca[legit_mask_train]
    
    return {
        'df_mexico': df_mexico,
        'feature_cols': feature_cols,
        'train_df': train_df,
        'test_df': test_df,
        'scaler': scaler,
        'pca': pca,
        'X_train_raw': X_train_raw,
        'y_train': y_train,
        'X_test_raw': X_test_raw,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'X_train_legit': X_train_legit,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }

if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("Datos cargados exitosamente:")
    print(f"- Muestras México: {len(data['df_mexico'])}")
    print(f"- Columnas numéricas predictoras: {len(data['feature_cols'])}")
    print(f"- Train: {len(data['X_train_pca'])} (Legítimas: {len(data['X_train_legit'])}, Fraudes: {data['y_train'].sum()})")
    print(f"- Test: {len(data['X_test_pca'])} (Legítimas: {(data['y_test'] == 0).sum()}, Fraudes: {data['y_test'].sum()})")
    print(f"- Varianza explicada 2 componentes: {data['cumulative_variance'][1]*100:.2f}%")
    print(f"- Varianza explicada 10 componentes: {data['cumulative_variance'][9]*100:.2f}%")
