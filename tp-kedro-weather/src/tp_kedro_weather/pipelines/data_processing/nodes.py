import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any
import matplotlib.pyplot as plt
import json


def load_weather_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Loads weather data from CSV.

    Args:
        weather_data: Raw weather data from CSV file.
    Returns:
        Weather DataFrame ready for processing.
    """
    return weather_data


def clean_weather_data(loaded_weather_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans weather data by converting columns to numeric and filling missing values.

    Args:
        loaded_weather_data: Raw weather data loaded from CSV.
    Returns:
        Cleaned weather data with numeric humidity and windspeed columns,
        missing values filled with column means.
    """
    # Créer une copie pour éviter de modifier les données originales
    df_cleaned = loaded_weather_data.copy()
    
    # Convertir humidity, windspeed ET temperature en numériques
    df_cleaned['humidity'] = pd.to_numeric(df_cleaned['humidity'], errors='coerce')
    df_cleaned['windspeed'] = pd.to_numeric(df_cleaned['windspeed'], errors='coerce')
    df_cleaned['temperature'] = pd.to_numeric(df_cleaned['temperature'], errors='coerce')
    
    # Remplacer les valeurs manquantes par la moyenne de chaque colonne
    df_cleaned['humidity'] = df_cleaned['humidity'].fillna(df_cleaned['humidity'].mean())
    df_cleaned['windspeed'] = df_cleaned['windspeed'].fillna(df_cleaned['windspeed'].mean())
    df_cleaned['temperature'] = df_cleaned['temperature'].fillna(df_cleaned['temperature'].mean())
    
    return df_cleaned


def train_model(donnees_nettoyees: pd.DataFrame) -> Dict[str, Any]:
    """Trains a linear regression model to predict temperature from humidity and windspeed.

    Args:
        donnees_nettoyees: Cleaned weather data with numeric columns.
    Returns:
        Dictionary containing the trained model and performance metrics.
    """
    # Définir les features et le target
    features = ['humidity', 'windspeed']
    target = 'temperature'
    
    # Vérifier que les colonnes existent
    if not all(col in donnees_nettoyees.columns for col in features + [target]):
        missing_cols = [col for col in features + [target] if col not in donnees_nettoyees.columns]
        raise ValueError(f"Colonnes manquantes dans les données: {missing_cols}")
    
    # Préparer les données
    X = donnees_nettoyees[features]
    y = donnees_nettoyees[target]
    
    # Supprimer les lignes avec des valeurs manquantes
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions sur le test
    y_pred = model.predict(X_test)
    
    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Retourner le modèle et les métriques séparément
    return {
        'trained_model': model,
        'metrics': {
            'mse': mse,
            'r2_score': r2,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
    }


def extract_model(model_results: Dict[str, Any]) -> Any:
    """Extracts the trained model from training results.
    
    Args:
        model_results: Dictionary containing model and metrics.
    Returns:
        The trained model.
    """
    return model_results['trained_model']


def extract_metrics(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts the metrics from training results.
    
    Args:
        model_results: Dictionary containing model and metrics.
    Returns:
        Dictionary containing performance metrics.
    """
    return model_results['metrics']


def convert_metrics_to_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Converts metrics to JSON-serializable format for Kedro Viz.
    
    Args:
        metrics: Dictionary containing performance metrics.
    Returns:
        JSON-serializable dictionary with metrics.
    """
    # Convertir tous les types numpy en types Python natifs
    json_metrics = {
        'mse': float(metrics['mse']),
        'r2_score': float(metrics['r2_score']),
        'n_train_samples': int(metrics['n_train_samples']),
        'n_test_samples': int(metrics['n_test_samples']),
        'model_performance': 'Good' if metrics['r2_score'] > 0.7 else 'Needs Improvement'
    }
    return json_metrics


def create_metrics_visualization(metrics: Dict[str, Any]):
    """Creates a visualization of the model metrics.
    
    Args:
        metrics: Dictionary containing performance metrics.
    Returns:
        Matplotlib figure for saving.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Métriques du Modèle de Prédiction Météo', fontsize=16, fontweight='bold')
    
    # Graphique 1: MSE
    ax1.bar(['MSE'], [metrics['mse']], color='lightcoral', alpha=0.7)
    ax1.set_title('Mean Squared Error (MSE)', fontweight='bold')
    ax1.set_ylabel('MSE')
    ax1.text(0, metrics['mse']/2, f"{metrics['mse']:.4f}", 
             ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Graphique 2: R² Score
    ax2.bar(['R² Score'], [metrics['r2_score']], color='lightgreen', alpha=0.7)
    ax2.set_title('Coefficient de Détermination (R²)', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)
    ax2.text(0, metrics['r2_score']/2, f"{metrics['r2_score']:.4f}", 
             ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Graphique 3: Échantillons
    samples = [metrics['n_train_samples'], metrics['n_test_samples']]
    labels = ['Train', 'Test']
    colors = ['skyblue', 'orange']
    ax3.pie(samples, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax3.set_title('Répartition Train/Test', fontweight='bold')
    
    # Graphique 4: Évaluation qualitative
    performance = 'Bon' if metrics['r2_score'] > 0.7 else 'À améliorer'
    color = 'green' if metrics['r2_score'] > 0.7 else 'red'
    ax4.text(0.5, 0.6, 'Performance du Modèle', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.4, performance, ha='center', va='center', 
             fontsize=20, fontweight='bold', color=color)
    ax4.text(0.5, 0.2, f'R² = {metrics["r2_score"]:.3f}', ha='center', va='center', 
             fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    return fig
