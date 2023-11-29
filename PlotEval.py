#!/bin/python3
#-*- coding:utf-8 -*-


# Importation des packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Importation des variables globales
from config import random_state


def plot_1d(features: pd.DataFrame, target: pd.Series) -> None:
	"""
	Réduit un DataFrame à une seule dimension en utilisant PCA et trace deux KDE plots en fonction des labels.

	Args:
		dataframe (pd.DataFrame): Le DataFrame contenant les features.
		labels (pd.Series): Une série contenant les labels correspondants aux features.

	Returns:
		None
	"""	
	# Réduire le DataFrame à une dimension en utilisant PCA
	pca = PCA(n_components=1, random_state=random_state)
	df_1d = pca.fit_transform(features)

	# Créez un DataFrame à partir de la réduction à une dimension et des labels
	df_1d = pd.DataFrame(df_1d, columns=["Dimension 1"])
	df_1d["Labels"] = target.reset_index(drop=True)

	# Tracez deux KDE plots en fonction des labels
	plt.figure(figsize=(8, 6))  # Ajustez la taille du plot si nécessaire
	sns.kdeplot(data=df_1d[df_1d["Labels"] == -1]["Dimension 1"], label="Défaites", fill=True)
	sns.kdeplot(data=df_1d[df_1d["Labels"] == 1]["Dimension 1"], label="Victoires", fill=True)
	plt.title("KDE Plot - Dimension 1")
	plt.xlabel("Valeur")
	plt.ylabel("Densité")
	plt.legend()
	plt.show()

def plot_2d(X: pd.DataFrame, y: pd.Series, method: str='pca') -> None:
	"""
	Effectue une analyse de réduction de dimension sur les données et génère un scatterplot avec légende.

	Cette fonction prend un DataFrame de données X et une série d'étiquettes y, effectue une réduction de dimension
	en 2 dimensions en utilisant la méthode spécifiée (PCA ou t-SNE), et affiche un scatterplot coloré en fonction
	des étiquettes.

	Args:
		X (pd.DataFrame): Les données à utiliser pour la réduction de dimension.
		y (pd.Series): Les étiquettes correspondantes.
		method (str, optional): La méthode de réduction de dimension à utiliser ('pca' ou 'tsne'). Par défaut, 'pca'.

	Returns:
		None
	"""	
	# Réduction de dimension en utilisant PCA
	if method == "pca":
		reduction_result = PCA(n_components=2, random_state=random_state).fit_transform(X)

	# Réduction de dimension en utilisant t-SNE
	elif method == "tsne":
		reduction_result = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=random_state).fit_transform(X)

	else:
		raise ValueError("La méthode spécifiée doit être 'pca' ou 'tsne'.")

	# Extraction des coordonnées x et z
	x = [i for i, j in reduction_result]
	z = [j for i, j in reduction_result]

	# Détermination des couleurs en fonction des étiquettes
	colors: List[str] = ["red" if label == 1 else "blue" for label in y.tolist()]

	# Affichage du scatterplot avec légende
	plt.figure(figsize=(12, 8))
	for label, color in {"Défaites": "blue", "Victoires": "red"}.items():
		plt.scatter([], [], c=color, label=label)
	plt.scatter(x, z, c=colors, marker='o', edgecolors="black")
	plt.xlabel("Features 1")
	plt.ylabel("Features 2")
	plt.legend(loc="best")
	plt.show()


def plot_confusion_matrix(y_true: Any, y_pred: Any) -> None:
	"""
	Affiche une matrice de confusion sous forme de heatmap.

	Args:
		y_true (Any): Les vraies étiquettes.
		y_pred (Any): Les étiquettes prédites.

	Returns:
		None
	"""
	# Calculer la matrice de confusion
	conf_matrix = confusion_matrix(y_true, y_pred)

	# Créer un heatmap de la matrice de confusion
	plt.figure(figsize=(12, 8))
	sns.set(font_scale=1.2)  # Réglez la taille de la police
	sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False,
				xticklabels=["Défaites", "Victoires"], yticklabels=["Défaites", "Victoires"])
	plt.xlabel("Prédictions")
	plt.ylabel("Réalités")
	plt.title("Matrice de Confusion")
	plt.show()

def plot_roc_auc(y_true: Any, y_pred: Any) -> None:
	"""
	Affiche la courbe ROC (Receiver Operating Characteristic) avec l'AUC (Area Under the Curve).

	Args:
		y_true (Any): Les vraies étiquettes.
		y_pred (Any): Les scores prédits (probabilités).

	Returns:
		None
	"""
	# Calculer la courbe ROC et l'AUC
	fpr, tpr,_ = roc_curve(y_true, y_pred)
	roc_auc = roc_auc_score(y_true, y_pred)

	# Tracer la courbe ROC
	plt.figure(figsize=(12, 8))
	plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Courbe ROC (AUC = {roc_auc:0.2f})")
	plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("Taux de Faux Positifs (FPR)")
	plt.ylabel("Taux de Vrais Positifs (TPR)")
	plt.title("Courbe ROC")
	plt.legend(loc="lower right")
	plt.show()

def evaluate_classification(y_true: Any, y_pred: Any, name_model: str='') -> None :
	"""
	Évalue la classification en affichant un rapport de classification, une matrice de confusion
	et une courbe ROC AUC.

	Args:
		y_true (array-like): Les vraies étiquettes de classe.
		y_pred (array-like): Les étiquettes de classe prédites.

	Returns:
		None
	"""
	# Rééquilibrer les poids des classes
	n_zero = sum([1 if i == -1 else 0 for i in y_true])
	n_one = sum([1 if i == 1 else 0 for i in y_true])
	if n_zero < n_one:
		weight_zero = n_one / n_zero
		weight_one = 1
	elif n_one < n_zero:
		weight_zero = 1
		weight_one = n_zero / n_one
	else:
		weight_zero = 1
		weight_one = 1
	
	# Créer la liste des poids pour chaque classe
	sample_weight = [weight_one if i == 1 else weight_zero for i in y_true]
	
	# Affiche le rapport de classification
	print(f"Rapport de classification {name_model} :\n\n{classification_report(y_true, y_pred, sample_weight=sample_weight)}")

	# Affiche la matrice de confusion
	plot_confusion_matrix(y_true, y_pred)

	# Affiche la courbe ROC AUC
	plot_roc_auc(y_true, y_pred)


def threshold_filter(reals: Any,
					 preds: Any,
					 elo_diffs: Any, 
					 probas: Any,
					 lower_threshold: float = 0.50,
					 upper_threshold: float = 0.50) -> pd.DataFrame:
	"""
	Filtre les résultats en fonction de seuils.

	Args:
		reals (array-like): Les vraies étiquettes de classe.
		preds (array-like): Les prédictions du modèle.
		elo_diffs (array-like): Les différences Elo.
		probas (array-like): Les probabilités prédites.
		lower_threshold (float, optional): Seuil inférieur pour le filtrage. Par défaut, 0.5.
		upper_threshold (float, optional): Seuil supérieur pour le filtrage. Par défaut, 0.5.

	Returns:
		pd.DataFrame: Un DataFrame contenant les résultats filtrés.
	"""    
	# Fonction pour calculer la probabilité originale
	compute_OriginalProba = lambda x: 1 / (1 + 10 ** (-x / 400))
	
	# Initialise le dataframe
	df = pd.DataFrame({"real": reals, "prediction": preds, "original_proba": elo_diffs, "adjusted_proba": probas})

	# Calcul de la probabilité originale
	df["original_proba"] = df["original_proba"].apply(func=compute_OriginalProba)
	
	# Filtrage des lignes
	df = df[(df["adjusted_proba"] <= lower_threshold) | (upper_threshold <= df["adjusted_proba"])]
	
	return df


def calculate_confidence_levels(df: pd.DataFrame, step: float, column: str) -> Tuple[pd.DataFrame, list]:
	"""
	Calcule le niveaux de confiance des prédictions en fonction de leur tranche de probabilité.

	Args:
		df (DataFrame): Le DataFrame contenant les données.
		step (float): Le pas entre les tranches de probabilités.
		column (str): Le nom de la colonne contenant les probabilités.

	Returns:
		Tuple[DataFrame, list]: Un tuple contenant deux éléments. Le premier élément est un DataFrame
		contenant les niveaux de confiance, et le deuxième élément est une liste des libellés des tranches.

	Example:
		Pour calculer les niveaux de confiance à partir d'un DataFrame 'df_final' avec un pas de 0.1
		et en utilisant la colonne 'adjusted_proba' :
		>>> confidence_levels, labels = calculate_confidence_levels(df_final, 0.1, "adjusted_proba")
	"""
	# Définir le minimum et le maximum avec arrondi personnalisé
	custom_round = lambda number: abs(round(number, 1)) if number < 0.50 else abs(round(number + 0.05, 1))
	min_ = custom_round(df[column].min())
	max_ = custom_round(df[column].max())

	# Définir les tranches (buckets) des probabilités
	buckets = np.arange(min_, max_, step)
	labels = [f"{i:.2f} - {i+step:.2f}" for i in np.arange(min_, max_ - step, step)]

	# Créer les tranches de probabilité
	df["proba_class"] = pd.cut(df[column], bins=buckets, labels=labels)

	# Initialiser la colonne 'confidence' en indiquant si la réalité et la prédiction sont correctes
	df["confidence"] = (df["real"] == df["prediction"]).astype(int)

	# Grouper les données en fonction des tranches de probabilités
	grouped = df.groupby(by="proba_class", observed=True)
	
	# Calculer la proportion de prédictions correctes pour chaque tranche
	confidence_levels = grouped["confidence"].mean().reset_index()
	
	# Calculer le nombre d'éléments de chaque tranche
	counts = df["proba_class"].value_counts().sort_index().reset_index()
	counts.columns = ["proba_class", "count"]

	# Fusionner les deux DataFrames sur la colonne "proba_class"
	confidence_levels = confidence_levels.merge(right=counts, on="proba_class")

	# Trier les données par 'proba_class'
	confidence_levels = confidence_levels.sort_values(by="proba_class")

	# Renvoyer le typle
	return confidence_levels, labels