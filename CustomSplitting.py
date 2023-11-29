#!/bin/python3
#-*- coding:utf-8 -*-


# Importation des modules
import pandas as pd
import numpy as np
from typing import Any, Union, Tuple
from twinning import twin


def train_test_split_twin(features: Any, target: Any, test_size: Union[int, float] = 0.2, random_state: int = None) -> Tuple[Any, Any, Any, Any]:
    """
    Divise un jeu de données en ensembles d'entraînement et de test en utilisant l'algorithme SPlit.

    Args:
    -----
        features (Any): Les caractéristiques (features) du jeu de données, pouvant être de type DataFrame, Series ou tableau NumPy.
        target (Any): La cible du jeu de données, pouvant être de type DataFrame, Series ou tableau NumPy.
        test_size (Union[int, float], optional): La taille de l'ensemble de test. Un nombre entier, ou une proportion. (default=0.2)
        random_state (int, optional) : La graine pour l'algorithme SPlit, contrôlant la répartition des données dans l'ensemble de test. (default=None)

    Returns:
        Tuple[Any, Any, Any, Any] : Un tuple contenant les ensembles d'entraînement et de test pour les caractéristiques et la cible.

    Remarks:
        - Si les données d'entrée sont des DataFrames ou des Series Pandas, elles seront converties en tableaux NumPy pour le traitement.
        - La valeur de `test_size` détermine la taille de l'ensemble de test en fonction des règles suivantes :
        - Si 0 < test_size < 1, il s'agit d'une proportion.
        - Si test_size >= 1, il s'agit d'un nombre absolu d'échantillons.
        - La valeur de `random_state` doit être comprise entre 0 et la longueur des caractéristiques moins 1.
        - La fonction Twin Test est utilisé pour répartir les données en ensembles d'entraînement et de test.

    Exemple:
        X_train, X_test, y_train, y_test = train_test_split_twin(features, target, test_size=0.2, random_state=42)
    """

    # Copie des caractéristiques (features) et de la cible (target)
    X = features.copy()
    y = target.copy()
    len_X = len(X)

    # Conversion en tableau NumPy si les entrées sont des DataFrames ou des Series Pandas
    if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
        X = X.to_numpy()
    if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
        y = y.to_numpy()
    
    # Calcul de la taille de l'ensemble de test en fonction de test_size
    if (0 < test_size) and (test_size < 1):
        r = int(1 / test_size)
    elif 1 <= test_size:
        r = int(1 / (test_size / len_X))
    else:
        raise ValueError("test_size doit être entre 0 et 1 ou entre 1 et len(features).")

    # Vérification de la validité de random_state
    if random_state is not None and (random_state < 0 or random_state >= len_X):
        raise ValueError("random_state doit être compris entre 0 et len(features)-1.")

    # Appel de la fonction twin en utilisant random_state comme paramètre u1
    test_indices = twin(data=np.column_stack(tup=(X, y)), r=r, u1=random_state)

    # Création de masques pour les ensembles d'entraînement et de test
    test_indices = [True if i in test_indices else False for i in range(0, len_X)]
    train_indices = [True if i == False else False for i in test_indices]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Conversion des données de sortie en DataFrame ou Series Pandas si nécessaire
    if isinstance(features, pd.DataFrame):
        X_train = pd.DataFrame(data=X_train, columns=features.columns)
        X_test = pd.DataFrame(data=X_test, columns=features.columns)
    elif isinstance(features, pd.Series):
        X_train = pd.Series(data=X_train, name=features.name)
        X_test = pd.Series(data=X_test, name=features.name)

    if isinstance(target, pd.DataFrame):
        y_train = pd.DataFrame(data=y_train, columns=target.columns)
        y_test = pd.DataFrame(data=y_test, columns=target.columns)
    elif isinstance(target, pd.Series):
        y_train = pd.Series(data=y_train, name=target.name)
        y_test = pd.Series(data=y_test, name=target.name)

    # Retourne les ensembles d'entraînement et de test
    return X_train, X_test, y_train, y_test