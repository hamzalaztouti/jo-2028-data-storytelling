# JO 2028 - Analyse et Prédiction des Performances Olympiques

## Contexte

Ce projet a été réalisé dans le cadre du module Data & IA (Bachelor 3).

L'objectif est de concevoir une application de data storytelling permettant d’analyser les performances historiques des Jeux Olympiques et de proposer une estimation des performances futures en vue des JO 2028.

Le projet couvre l’ensemble des étapes d’un pipeline de data science : préparation des données, analyse exploratoire, modélisation et déploiement d’une application interactive.

## Objectifs

- Analyser les performances des Jeux Olympiques
- Identifier les tendances par pays, sport et genre
- Mettre en évidence les variables influençant la performance
- Construire un modèle de machine learning
- Développer une application interactive de visualisation et de prédiction

## Données

Dataset utilisé : `athlete_events.csv`

Contenu du dataset :
- Informations sur les athlètes : âge, taille, poids
- Pays (NOC)
- Sport et discipline
- Participation aux Jeux Olympiques
- Médailles obtenues (Or, Argent, Bronze)

## Méthodologie

### 1. Data Audit
- Analyse des valeurs manquantes
- Détection des doublons
- Statistiques descriptives
- Identification des valeurs aberrantes

### 2. Nettoyage des données
- Conversion des types de variables
- Suppression ou gestion des valeurs manquantes
- Préparation des données pour l’analyse

### 3. Feature Engineering
- Création de la variable cible `Has_Medal`
- Sélection des variables pertinentes pour la modélisation

### 4. Analyse exploratoire
- Analyse des pays les plus représentés
- Répartition des athlètes par sexe
- Évolution des médailles dans le temps
- Analyse des corrélations entre variables physiques

### 5. KPI
- Nombre total d’athlètes
- Nombre total de pays
- Nombre total de sports
- Nombre total de médailles

## Modélisation

Deux modèles ont été étudiés :
- Logistic Regression
- Random Forest

Le modèle Random Forest a été retenu pour la prédiction finale en raison de ses bonnes performances sur les données.

Variables utilisées :
- Age
- Height
- Weight

Variable cible :
- Has_Medal

## Évaluation

Les performances du modèle ont été évaluées avec plusieurs métriques :
- Accuracy
- Matrice de confusion
- ROC Curve
- AUC

Le modèle retenu présente de bonnes performances globales pour une première estimation.

## Application Streamlit

Une application interactive a été développée avec Streamlit pour présenter les résultats du projet.

### Fonctionnalités principales

#### Analyse
- Affichage des KPI
- Graphiques interactifs
- Filtres dynamiques par année, sport, sexe, saison et pays

#### Carte
- Visualisation mondiale des médailles par pays

#### Prédiction
- Simulation d’un profil d’athlète
- Estimation de la probabilité d’obtenir une médaille
- Visualisation de l’importance des variables du modèle

## Technologies utilisées

- Python
- Pandas
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook

## Structure du projet

jo-2028-data-storytelling/
│
├── data/
│   └── athlete_events.csv
│
├── app.py
├── projet_jo_2028.ipynb
├── README.md
├── requirements.txt
├── .gitignore


## Auteurs

- **Hamza Laztouti**
- **Rossaina Tahiri**

Bachelor 3 Data & IA  
Ynov Campus