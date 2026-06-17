# JO 2028 - Analyse, Segmentation et Prédiction des Performances Olympiques

## Contexte

Ce projet a été réalisé dans le cadre du Bachelor 3 Data & IA à Ynov Campus.

L'objectif est de développer une application complète de Data Storytelling permettant d'explorer l'historique des Jeux Olympiques, d'identifier les tendances majeures, de segmenter les profils d'athlètes grâce au Machine Learning et de proposer des projections pour les Jeux Olympiques de Los Angeles 2028.

Le projet couvre l'ensemble du cycle de la Data Science :

* Collecte et compréhension des données
* Nettoyage et préparation
* Analyse exploratoire
* Visualisation interactive
* Segmentation des athlètes (K-Means)
* Modélisation prédictive
* Déploiement avec Streamlit

---

# Objectifs

* Analyser l'évolution historique des Jeux Olympiques
* Identifier les pays les plus performants
* Étudier l'influence du sexe, de l'âge et des caractéristiques physiques
* Explorer les performances des athlètes et disciplines
* Segmenter les athlètes selon leurs profils sportifs
* Prédire les performances futures en vue des JO 2028
* Concevoir une application interactive et intuitive

---

# Dataset

Dataset utilisé :

athlete_events.csv

Source :

https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results

Le dataset contient :

* Nom de l'athlète
* Sexe
* Âge
* Taille
* Poids
* Pays (NOC)
* Sport
* Épreuve
* Saison (Summer / Winter)
* Année
* Médaille obtenue

Le projet utilise les données historiques jusqu'aux Jeux Olympiques de Rio 2016.

---

# Préparation des données

## Nettoyage

* Suppression des doublons
* Gestion des valeurs manquantes
* Conversion des types de données
* Vérification de la cohérence des variables

## Feature Engineering

Création des variables :

* Has_Medal
* Participation_Count
* Total_Medals_Athlete
* Country_Medals

Ces variables sont utilisées pour les analyses et les modèles de Machine Learning.

---

# Analyse Exploratoire

L'application permet :

## KPI globaux

* Nombre total d'athlètes
* Nombre total de pays
* Nombre total de sports
* Nombre total de médailles

## Analyses réalisées

* Top pays médaillés
* Répartition des médailles par sexe
* Répartition été / hiver
* Analyse des disciplines olympiques
* Analyse des performances historiques
* Corrélations entre variables physiques

---

# Carte Mondiale des Médailles

Une carte interactive permet de visualiser :

* Les pays médaillés
* Les performances mondiales
* Les médailles d'or, d'argent ou de bronze

Filtres disponibles :

* Saison
* Année
* Type de médaille

Le filtre des années s'adapte automatiquement selon la saison sélectionnée :

* Summer → années des JO d'été uniquement
* Winter → années des JO d'hiver uniquement

---

# Performances et Athlètes

Cette section présente :

## Classement des athlètes

* Athlètes les plus médaillés
* Athlètes les plus performants

## Nouvelles générations

Identification des jeunes athlètes à fort potentiel.

## Cotes par discipline

Calcul d'un score pondéré prenant en compte :

* Les médailles
* La régularité
* Les participations

---

# Timeline Historique

Analyse temporelle des Jeux Olympiques :

* Évolution du nombre de médailles
* Évolution des disciplines
* Comparaison JO d'été / JO d'hiver
* Analyse des records historiques

Les visualisations sont limitées aux données disponibles jusqu'en 2016.

---

# Clustering K-Means

Une approche de Machine Learning non supervisé a été utilisée afin de segmenter les athlètes.

Variables utilisées :

* Age
* Height
* Weight
* Participation_Count
* Total_Medals_Athlete

Méthodologie :

* Standardisation des données
* Méthode du coude
* K optimal = 3

Profils obtenus :

### Débutants

Athlètes ayant peu de participations et peu de médailles.

### Intermédiaires

Athlètes expérimentés avec performances régulières.

### Performants

Athlètes ayant obtenu le plus grand nombre de médailles et de participations.

---

# Modélisation Prédictive

Deux modèles ont été évalués :

## Logistic Regression

Modèle linéaire servant de référence.

## Random Forest

Modèle retenu pour les prédictions finales.

Variables utilisées :

* Age
* Height
* Weight
* Participation_Count
* Total_Medals_Athlete
* Country_Medals

Variable cible :

* Has_Medal

---

# Évaluation des Modèles

Les modèles ont été évalués à l'aide de :

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* AUC
* Matrice de confusion

Le modèle Random Forest a obtenu les meilleurs résultats et a été retenu pour l'application finale.

---

# Prédictions JO 2028

L'application propose :

## Simulation d'un athlète

Prédiction de la probabilité d'obtenir une médaille.

## Projection des pays favoris

Analyse des tendances historiques afin d'estimer les performances potentielles des pays lors des Jeux Olympiques de Los Angeles 2028.

Les projections utilisent les quatre dernières olympiades disponibles dans le dataset historique.

---

# Application Streamlit

L'application est composée de six modules :

1. Analyse
2. Carte
3. Athlètes
4. Timeline
5. K-Means
6. Prédiction 2028

Fonctionnalités :

* Interface moderne
* Filtres dynamiques
* Graphiques interactifs Plotly
* Cartes géographiques
* Segmentation Machine Learning
* Modèles prédictifs

---

# Technologies Utilisées

* Python
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Streamlit
* Jupyter Notebook

---

# Structure du Projet

```text
jo-2028-data-storytelling/
│
├── data/
│   └── athlete_events.csv
│
├── app.py
├── JO_OLMP_2028.ipynb
├── README.md
├── requirements.txt
├── .gitignore
```

---

# Auteurs

Hamza Laztouti
Rossaina Tahiri

Bachelor 3 Data & IA

Ynov Campus

Projet Fil Rouge 2025-2026
