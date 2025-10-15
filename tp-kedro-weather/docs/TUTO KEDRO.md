## **TUTO KEDRO**

## **Objectif du projet**

**Structurer un projet avec Kedro**

* Créer des pipelines pour automatiser les étapes de traitement des données et de modélisation.

* Définir et utiliser un catalogue de datasets pour gérer les fichiers de manière claire.

**Gérer les données avec Kedro**

* Charger un jeu de données contenant des valeurs manquantes et des incohérences.

* Nettoyer et transformer les données dans des nœuds Kedro dédiés.

**Entraîner et évaluer un modèle ML**

* Définir un pipeline de modélisation pour prédire la **température** à partir de `humidity` et `windspeed`.

* Calculer et sauvegarder les métriques de performance dans Kedro.

**Sauvegarder les résultats avec Kedro**

* Stocker le modèle entraîné dans un dataset dédié.

* Garantir la reproductibilité et la traçabilité des expériences.

### **Étape 0 : Préparer l’environnement**

1. Créer un environnement virtuel (recommandé pour isoler le projet) :

   `python -m venv venv`

2. Activer l’environnement virtuel :

* Windows :

  `venv\Scripts\activate`


* Mac/Linux :

  `source venv/bin/activate`

3. Installer les packages nécessaires :

`pip install kedro pandas numpy scikit-learn kedro-viz pyspark` 

### **Étape 1 : Créer le projet Kedro**

1. Ouvrir un terminal et créer un nouveau projet Kedro :  
    `kedro new`  
2. Donner un nom au projet (ex : `tp_kedro_weather`) et suivre les instructions de Kedro.

3. Vérifier que la structure du projet a été créée correctement :  
   `conf/`  
   `data/`  
   `src/`  
     
   

### **Étape 2 : Charger les données avec un Node**

1. Placer le fichier CSV (`weather_data.csv`) dans `data/01_raw/`.

2. Définir un **dataset** dans `conf/base/catalog.yml`

3. Créer un **node Kedro** pour charger les données :
  - Fonction `load_weather_data()` qui lit le CSV avec `pandas`.
  - Retourne un DataFrame pour le pipeline.

### **Étape 3 : Nettoyer les données**

1. Créer un node `clean_weather_data(df)` :

   * Convertir les colonnes `humidity` et `windspeed` en numériques (`pd.to_numeric(errors='coerce')`).

   * Remplacer les valeurs manquantes par la moyenne de chaque colonne.

2. Sortie : `donnees_nettoyees`, qui sera utilisée pour l’entraînement du modèle.

### **Étape 4 : Entraîner un modèle**

1. Créer un node `train_model(df)` :

   * Utiliser `humidity` et `windspeed` comme **features**.

   * Utiliser `temperature` comme **target**.

   * Diviser les données en train/test et entraîner un modèle de régression linéaire.

2. Sorties :

   * `trained_model` : le modèle entraîné.

   * `metrics` : performance du modèle (ex. MSE).

### **Étape 5 : Sauvegarder les résultats**

1. Définir des datasets dans le catalogue pour le modèle et les métriques :

    `Modele_entraine:`  
     `type: pickle.PickleDataset`  
     `filepath: data/04_models/modele.pkl`  
   `metrics:`  
     `type: pickle.PickleDataset`  
     `filepath: data/04_models/metrics.pkl`  
2. Ajouter les nodes correspondants au pipeline pour sauvegarder ces fichiers.
