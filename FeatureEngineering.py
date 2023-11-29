#!/bin/python3
#-*- coding:utf-8 -*-


# Importation des packages
import pandas as pd
from typing import Tuple


def create_features_and_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Crée les features et les labels à partir d'un DataFrame de données.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        features (pd.DataFrame): Les features.
        labels (pd.Series): Les labels.
    """
    # Crée un DataFrame pour stocker les features
    features = pd.DataFrame()

    # Ajout des colonnes d'elos
    features["elo_h"] = data["elo_h"]
    features["elo_a"] = data["elo_a"]
    features["elo_diff"] = features["elo_h"] - features["elo_a"]

    # Ajout des colonnes des probas selon la loi de Laplace
    features["laplace_proba_h"] = data["laplace_proba_h"]
    features["laplace_proba_a"] = data["laplace_proba_a"]
    features["laplace_proba_diff"] = features["laplace_proba_h"] - features["laplace_proba_a"]

    # Ajout des colonnes des plus-minus
    features["sum_previous_pm_h"] = data["sum_previous_pm_h"]
    features["sum_previous_pm_a"] = data["sum_previous_pm_a"]
    features["sum_previous_pm_diff"] = features["sum_previous_pm_h"] - features["sum_previous_pm_a"]

    # Ajout des colonnes des scores streak
    features["streak_h"] = data["streak_h"]
    features["streak_a"] = data["streak_a"]
    features["streak_diff"] = features["streak_h"] - features["streak_a"]
    
    # Ajout des classements précédents
    features["lsr_h"] = data["lsr_h"]
    features["lsr_a"] = data["lsr_a"]
    features["lsr_diff"] = features["lsr_h"] - features["lsr_a"]

    # Les labels sont la colonne 'win'
    labels = data["win"].apply(func=lambda x: 1 if x == 1 else -1)

    # Utiliser les dates comme index
    features.index = pd.to_datetime(arg=data["date"].values)
    labels.index = pd.to_datetime(arg=data["date"].values)

    # Retourner les résultats
    return features, labels