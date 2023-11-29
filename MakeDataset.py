#!/bin/python3
#-*- coding:utf-8 -*-


# Importation des packages
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


# Définition des saisons sous forme de liste de chaînes de caractères
SEASONS: List[str] = [str(i) for i in range(2004, 2024, 1)]

# Définition des séparateurs pour les équipes à domicile et à l'extérieur
HOME_SEP: str = "vs."
AWAY_SEP: str = "@"

# Colonnes pour les équipes à domicile
HOME_COLS: List[str] = [
	"season", "date", "win", "min", "fgm_h", "fga_h", "fg_pct_h", "fg3m_h", "fg3a_h", "fg3_pct_h", 
	"ftm_h", "fta_h", "ft_pct_h", "oreb_h", "dreb_h", "reb_h", "ast_h", "stl_h", "blk_h", "tov_h", 
	"pf_h", "pts_h", "pm_h", "home", "away"
]

# Colonnes pour les équipes à l'extérieur
AWAY_COLS: List[str] = [
	"season", "date", "win_a", "min", "fgm_a", "fga_a", "fg_pct_a", "fg3m_a", "fg3a_a", "fg3_pct_a", 
	"ftm_a", "fta_a", "ft_pct_a", "oreb_a", "dreb_a", "reb_a", "ast_a", "stl_a", "blk_a", "tov_a", 
	"pf_a", "pts_a", "pm_a", "home", "away"
]

# Ordre final des colonnes dans le DataFrame
FINAL_ORDER: List[str] = [
	"season", "date", "win", "min", "home", "away", "pts_h", "ast_h", "reb_h", "blk_h", "stl_h", 
	"tov_h", "fgm_h", "fga_h", "fg_pct_h", "fg3m_h", "fg3a_h", "fg3_pct_h", "ftm_h", "fta_h", 
	"ft_pct_h", "oreb_h", "dreb_h", "pf_h", "pm_h", "pts_a", "ast_a", "reb_a", "blk_a", "stl_a", 
	"tov_a", "fgm_a", "fga_a", "fg_pct_a", "fg3m_a", "fg3a_a", "fg3_pct_a", "ftm_a", "fta_a", 
	"ft_pct_a", "oreb_a", "dreb_a", "pf_a", "pm_a"
]

# Champions nba de chaque saison
CHAMPIONS: Dict[int, str] = {
	2004: "DET", 2005: "SAS", 2006: "MIA", 2007: "SAS",
	2008: "BOS", 2009: "LAL", 2010: "LAL", 2011: "DAL",
	2012: "MIA", 2013: "MIA", 2014: "SAS", 2015: "GSW",
	2016: "CLE", 2017: "GSW", 2018: "GSW", 2019: "TOR",
	2020: "LAL", 2021: "MIL", 2022: "GSW", 2023: "DEN"
}


def read_json(filename: str, season: str) -> pd.DataFrame:
	"""
	Lit un fichier JSON et crée un DataFrame Pandas à partir de ses données.

	Args:
		filename (str): Le nom du fichier JSON à lire.
		season (str): L'année de la saison associée aux données.

	Returns:
		pd.DataFrame: Un DataFrame Pandas contenant les données lues à partir du fichier JSON.
	"""
	# Ouvre le fichier JSON en mode lecture
	with open(file=filename, mode='r') as box:
		# Charge les données JSON dans une variable
		data = json.load(fp=box)

	# Récupère les en-têtes des colonnes
	headers = data["resultSets"][0]["headers"]

	# Récupère les lignes de données
	rows = data["resultSets"][0]["rowSet"]

	# Crée un DataFrame Pandas à partir des en-têtes et des lignes de données
	data = pd.DataFrame(data=rows, columns=headers)

	# Ajoute la colonne 'SEASON_ID' contenant l'année de la saison
	data["SEASON_ID"] = season

	# Supprime les colonnes indésirables du DataFrame
	data = data.drop(labels=["TEAM_ID", "GAME_ID", "VIDEO_AVAILABLE", "TEAM_ABBREVIATION", "TEAM_NAME"], axis=1)

	# Retourne le DataFrame final
	return data

def split_matchup(df: pd.DataFrame, new_cols: List[str], sep: str) -> pd.DataFrame:
	"""
	Sépare la colonne 'MATCHUP' d'un DataFrame en colonnes 'home' et 'away' en fonction d'un séparateur spécifié.

	Args:
		df (pd.DataFrame): Le DataFrame contenant la colonne 'MATCHUP' à diviser.
		new_cols (List[str]): Une liste contenant les noms des nouvelles colonnes ('date', 'min', 'home', 'away').
		sep (str): Le séparateur utilisé pour diviser les valeurs de la colonne 'MATCHUP' (par exemple, "vs." ou "@").

	Returns:
		pd.DataFrame: Un nouveau DataFrame avec les colonnes 'date', 'min', 'home' et 'away'.
	"""
	# Crée un DataFrame 'res' en filtrant les lignes contenant le séparateur spécifié dans la colonne 'MATCHUP'
	res = df[df["MATCHUP"].str.contains(sep)].copy()

	if sep == HOME_SEP:
		# Fonction pour extraire le nom de l'équipe à domicile
		filter_home = lambda home: home.split(sep=sep)[0].strip()
		# Fonction pour extraire le nom de l'équipe à l'extérieur
		filter_away = lambda away: away.split(sep=sep)[1].strip()
	elif sep == AWAY_SEP:
		# Fonction pour extraire le nom de l'équipe à domicile
		filter_home = lambda home: home.split(sep=sep)[1].strip()
		# Fonction pour extraire le nom de l'équipe à l'extérieur
		filter_away = lambda away: away.split(sep=sep)[0].strip()
	else:
		# Si le séparateur n'est pas "vs." ou "@", lève une exception
		raise ValueError("Séparateur non valide")

	# Applique les fonctions de filtre aux valeurs de la colonne 'MATCHUP' pour créer les colonnes 'home' et 'away'
	res["home"] = res["MATCHUP"].apply(func=filter_home)
	res["away"] = res["MATCHUP"].apply(func=filter_away)

	# Supprime la colonne 'MATCHUP'
	res = res.drop(labels=["MATCHUP"], axis=1)

	# Renomme les colonnes selon les noms spécifiés dans 'new_cols'
	res.columns = new_cols

	# Trie le DataFrame par 'date', 'home' et 'away', puis réinitialise les index
	res = res.sort_values(by=["date", "home", "away"], ascending=True).reset_index(drop=True)

	return res

def make_seasonDF(season: str) -> pd.DataFrame:
	"""
	Crée un DataFrame saison à partir des données JSON pour une saison NBA spécifiée.

	Args:
		season (str): La saison NBA spécifiée (par exemple, "2023").

	Returns:
		pd.DataFrame: Un DataFrame contenant les données de la saison spécifiée.
	"""
	# Lecture des données JSON pour la saison spécifiée
	df = read_json(filename=f"./dataJson/nba_BS_{season}.json", season=season)

	# Séparation des données pour les équipes à domicile
	home = split_matchup(df=df, new_cols=HOME_COLS, sep=HOME_SEP)

	# Séparation des données pour les équipes à l'extérieur
	away = split_matchup(df=df, new_cols=AWAY_COLS, sep=AWAY_SEP)

	# Concaténation des DataFrames 'home' et 'away', en excluant certaines colonnes de 'away'
	final_df = pd.concat(objs=[home, away[[col for col in AWAY_COLS if col not in ["season", "win_a", "date", "min", "home", "away"]]]], axis=1)

	# Transformation des valeurs de la colonne 'win' en 1 pour 'W' (victoire) et 0 pour les autres valeurs
	final_df["win"] = final_df["win"].apply(lambda x: 1 if x == "W" else 0)

	# Réorganisation des colonnes du DataFrame final selon l'ordre spécifié dans 'FINAL_ORDER'
	return final_df[FINAL_ORDER]


def compute_expected_proba(elo_team: float, elo_oppo: float) -> float:
	"""
	Calcule la probabilité attendue de victoire d'une équipe en fonction de son score Elo et celui de l'adversaire.

	Args:
		elo_team (float): Le score Elo de l'équipe pour laquelle vous calculez la probabilité.
		elo_oppo (float): Le score Elo de l'adversaire de l'équipe.

	Returns:
		float: La probabilité attendue de victoire de l'équipe (comprise entre 0 et 1).
	"""
	# Utilisation de la formule Elo standard pour calculer la probabilité attendue
	# La formule est basée sur la différence de scores Elo entre les équipes.
	# Plus la différence est grande, moins la probabilité d'une victoire est élevée.
	return 1 / (1 + 10 ** ((elo_oppo - elo_team) / 400))

def update_elos(elo_winner: float, elo_loser: float, f: float) -> Tuple[float, float]:
	"""
	Met à jour les scores Elo des équipes en fonction des performances d'un match.

	Cette fonction prend en compte les scores Elo de l'équipe gagnante et de l'équipe perdante,
	ainsi qu'un facteur bonus pour le type de match.

	Args:
		elo_winner (float): Le score Elo de l'équipe gagnante.
		elo_loser (float): Le score Elo de l'équipe perdante.
		f (float): Facteur bonus pour le type de match (1 pour match à domicile, 2 pour match à l'extérieur).

	Returns:
		Tuple[float, float]: Un tuple contenant les nouveaux scores Elo de l'équipe gagnante et de l'équipe perdante.
	"""
	# Facteur K pour les mises à jour Elo
	k_factor = 20

	# L'avantage à domicile
	h_factor = 75

	# Calcul de la probabilité attendue de victoire pour l'équipe gagnante en utilisant les scores Elo
	if f == 1:
		expected_proba = compute_expected_proba(elo_winner+h_factor, elo_loser-h_factor)
	else:
		expected_proba = compute_expected_proba(elo_winner-h_factor, elo_loser+h_factor)

	# Mise à jour des scores Elo pour l'équipe gagnante et la perdante en fonction des résultats du match
	elo_winner = elo_winner + (k_factor  * f * (1 - expected_proba))
	elo_loser = elo_loser + (k_factor * -f * (1 - expected_proba))

	return elo_winner, elo_loser

def calculate_elos(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calcule et met à jour les scores Elo des équipes pour chaque match dans le DataFrame.

	Args:
		data (pd.DataFrame): DataFrame contenant les informations sur les matchs.

	Returns:
		pd.DataFrame: Le DataFrame mis à jour avec les scores Elo mis à jour après chaque match.
	"""
	# Trier le dataframe
	data = data.sort_values(by=["date"]).reset_index(drop=True)

	# Score Elo initial pour chaque équipe
	base_elo = 1500

	# Définir le bonus/pénalitée en fonction du lieux de rencontre
	f_h, f_a = 1, 0.5875/0.4125

	# Créer un dictionnaire pour stocker les scores Elo actuels de chaque équipe
	elos: Dict[str, float] = {team: base_elo for team in data["home"].unique()}

	# Définir la saison actuelle
	current_season = data.at[0, "season"]

	for index, row in data.iterrows():
		# Récupération des infos
		home, away, season, win = row["home"], row["away"], row["season"], row["win"]

		# Vérifier si la saison a changé
		if season != current_season:
			# Réinitialiser les scores Elo pour la nouvelle saison
			for team, elo in elos.items():
				elos[team] = (elo * 0.75) + (0.25 * base_elo)

			# Récupérer l'équipe championne et l'attribuer un bonus de points en début de nouvelles saison
			champ = CHAMPIONS[int(current_season)]
			elos[champ] += 150

			# Changer de saison
			current_season = season

		# Obtenir les scores Elo actuels des équipes
		elo_h, elo_a = elos[home], elos[away]

		# Mettre à jour le DataFrame avec les nouveaux scores Elo
		data.at[index, "elo_h"] = elo_h
		data.at[index, "elo_a"] = elo_a

		# Mettre à jour les scores Elo en fonction du résultat du match
		if win == 1:
			elo_h, elo_a = update_elos(elo_winner=elo_h, elo_loser=elo_a, f=f_h)
		else:
			elo_a, elo_h = update_elos(elo_winner=elo_a, elo_loser=elo_h, f=f_a)

		# Mettre à jour les scores Elo dans le dictionnaire
		elos[home], elos[away] = elo_h, elo_a

	return data


def compute_laplace_proba(events: List[int]) -> float:
	"""
	Calcule la probabilité de gagner selon la loi de succession de Laplace, en fonction du nombre de victoires et du total de parties.

	Args:
		events (List[int]): Une liste contenant le nombre de victoires et le nombre total de parties.

	Returns:
		float: La probabilité de gagner selon la loi de succession de Laplace, défini comme (victoires + 1) / (parties totales + 2).
	"""
	return (events[0] + 1) / (events[1] + 2)

def update_laplace_proba(events_home: List[int], events_away: List[int], win: int) -> Tuple[List[int], List[int]]:
	"""
	Met à jour la liste des évènements, pour les équipes à domicile et à l'extérieur en fonction du résultat du match.

	Args:
		events_home (List[int]): Une liste contenant le nombre de victoires et le nombre total de parties pour l'équipe à domicile.
		events_away (List[int]): Une liste contenant le nombre de victoires et le nombre total de parties pour l'équipe à l'extérieur.
		win (int) : Résultat du match, 1 si l'équipe à domicile gagne, 0 sinon.

	Returns:
		Tuple[List[int], List[int]]: Les listes mis à jour, pour l'équipe à domicile et celle à l'extérieur.
	"""
	# Incrémentation des victoires et des matchs joués en fonction de 'win'
	if win == 1:
		# Incrémentation des victoires pour l'équipe à domicile
		events_home[0] += 1
	else:
		# Incrémentation des victoires pour l'équipe à l'extérieur
		events_away[0] += 1

	# Incrémentation des matchs joués pour l'équipe à domicile
	events_home[1] += 1

	# Incrémentation des matchs joués pour l'équipe à l'extérieur
	events_away[1] += 1 

	# Retourne le tuple
	return events_home, events_away

def calculate_laplace_proba(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calcule la probabilité de gagner selon la loi de succession de Laplace pour chaque équipe au fil de chaque saison.

	Args:
		data (pd.DataFrame): DataFrame contenant les données triées par saison et date.

	Returns:
		pd.DataFrame: Le DataFrame d'origine avec les colonnes supplémentaires 'laplace_proba_h' et 'laplace_proba_a'.
	"""
	# Trier le dataframe
	data = data.sort_values(by=["date"]).reset_index(drop=True)

	# Récupérer les équipes
	teams = data["home"].unique()

	# Dictionnaire pour stocker les évènements des équipes
	events = {team: [0, 0] for team in teams}

	# Initialiser la saison actuelle
	current_season = data.at[0, "season"]

	# Itérer les dataframe par ligne
	for index, row in data.iterrows():
		# Récupération des infos nécessaires
		home, away, season, win = row["home"], row["away"], row["season"], row["win"]

		# Mettre à jour la saison si nécessaire
		if season != current_season:
			events = {team: [0, 0] for team in teams}
			current_season = season

		# Calcul du la proba selon la loi de Laplace pour l'équipe à domicile (laplace_proba_h) et à l'extérieur (laplace_proba_a)
		data.at[index, "laplace_proba_h"] = compute_laplace_proba(events=events[home])
		data.at[index, "laplace_proba_a"] = compute_laplace_proba(events=events[away])

		# Mettre à jour les évenenments
		events[home], events[away] = update_laplace_proba(events_home=events[home], events_away=events[away], win=win)

	return data


def compute_sum_pm(plus_minus: List[int]) -> float:
	"""
	Calcule la somme des plus-minus.

	Args:
		plus_minus (List[int]): Une liste contenant les plus minus.

	Returns:
		float: La somme des plus-minus, si la liste est non-vide, 0 sinon.
	"""
	# Retourner le résultat
	return sum(plus_minus) if len(plus_minus) != 0 else 0

def update_lists_pm(pm_home: List[int], pm_away: List[int], win: int, pm: int, window: int) -> Tuple[List[int], List[int]]:
	"""
	Met à jour la liste des plus-minus, pour les équipes à domicile et à l'extérieur en fonction du résultat du match.

	Args:
		pm_home (List[int]): Une liste contenant les plus-minus pour l'équipe à domicile.
		pm_away (List[int]): Une liste contenant les plus-minus pour l'équipe à l'extérieur.
		win (int) : Résultat du match, 1 si l'équipe à domicile gagne, 0 sinon.
		pm (int) : Le plus-minus du match.

	Returns:
		Tuple[List[int], List[int]]: Les listes des plus-minus mis à jour, pour l'équipe à domicile et celle à l'extérieur.
	"""
	# Vérifie si la longueur de la liste pm_home est égale à la taille de la fenêtre
	if len(pm_home) == window:
		# Si c'est une victoire
		if win == 1:
			# Retire le premier élément et ajoute +pm
			pm_home = pm_home[1:] + [+pm]
		else:
			# Sinon, Retire le premier élément et ajoute -pm
			pm_home = pm_home[1:] + [-pm]
	else:
		if win == 1:
			# Ajoute +pm à la fin de la liste
			pm_home = pm_home + [+pm]
		else:
			# Ajoute -pm à la fin de la liste
			pm_home = pm_home + [-pm]

	# Vérifie si la longueur de la liste pm_away est égale à la taille de la fenêtre
	if len(pm_away) == window:
		# Si c'est une victoire
		if win == 1:
			# Retire le premier élément et ajoute -pm
			pm_away = pm_away[1:] + [-pm]
		else:
			# Sinon, Retire le premier élément et ajoute +pm
			pm_away = pm_away[1:] + [+pm]
	else:
		if win == 1:
			# Ajoute -pm à la fin de la liste
			pm_away = pm_away + [-pm]
		else:
			# Ajoute +pm à la fin de la liste
			pm_away = pm_away + [+pm]

	# Retourne les listes mise à jour
	return pm_home, pm_away

def calculate_plus_minus(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calcule la somme des n précédents plus-minus de chaque équipe au fil de chaque saison.

	Args:
		data (pd.DataFrame): DataFrame contenant les données triées par saison et date.

	Returns:
		pd.DataFrame: Le DataFrame d'origine avec les colonnes supplémentaires 'sum_previous_pm_h' et 'sum_previous_pm_a'.
	"""
	# Trier le dataframe
	data = data.sort_values(by=["date"]).reset_index(drop=True)

	# Récupérer les équipes
	teams = data["home"].unique()

	# Dictionnaire pour stocker les plus-minus
	plus_minus = {team: [] for team in teams}

	# Définition de la fenêtre
	window = 5

	# Initialiser la saison actuelle
	current_season = data.at[0, "season"]

	# Itérer les dataframe par ligne
	for index, row in data.iterrows():
		# Récupération des infos nécessaires
		home, away, season, win = row["home"], row["away"], row["season"], row["win"]

		# Récupérer le plus-minus du match
		pm = abs(row["pm_h"])

		# Mettre à jour la saison si nécessaire
		if season != current_season:
			plus_minus = {team: [] for team in teams}
			current_season = season

		# Calcul de la somme des plus-minus, pour l'équipe à domicile et pour celle à l'extérieure
		data.at[index, "sum_previous_pm_h"] = compute_sum_pm(plus_minus=plus_minus[home])
		data.at[index, "sum_previous_pm_a"] = compute_sum_pm(plus_minus=plus_minus[away])

		# Mettre à jour les plus-minus
		plus_minus[home], plus_minus[away] = update_lists_pm(pm_home=plus_minus[home], pm_away=plus_minus[away], win=win, pm=pm, window=window)

	return data


def compute_streak_score(seq: str) -> float:
	"""
	Calcule le score de streak à partir d'une séquence de résultats précédents.

	Args:
		seq (str): Une séquence de résultats précédents sous forme de chaîne de caractères (ex : "hahha").

	Returns:
		float: Le score de streak de la séquence.
	"""
	# Vérifier si on a une séquence
	if len(seq) == 0:
		return 0

	# Définition des valeurs possibles
	events = {'h': 1, 'i': -2, 'a': 2, 'b': -1}

	# Initialisation du score
	score = 0

	# Calcul du score
	for event in seq:
		score += events[event]

	# Renvoi le score
	return score

def update_streaks(seq_home: str, seq_away: str, win: int, window: int) -> Tuple[str, str]:
	"""
	Met à jour les séquences de victoires consécutives pour les équipes à domicile et à l'extérieur en fonction du résultat du match.

	Args:
		seq_home (str): Séquence actuelle de victoires consécutives pour l'équipe à domicile.
		seq_away (str): Séquence actuelle de victoires consécutives pour l'équipe à l'extérieur.
		win (int): Résultat du match, 1 si l'équipe à domicile gagne, 0 sinon.
		window (int): Longueur de la fenêtre pour les séquences de victoires consécutives.

	Returns:
		Tuple[str, str]: Les séquences mises à jour pour l'équipe à domicile et à l'extérieur.
	"""
	# Récupérer les longueurs de séquences
	len_home = len(seq_home)
	len_away = len(seq_away)

	# Mettre à jour la séquence en fonction des conditions
	if len_home == window and len_away == window:
		if win == 1:
			seq_home = seq_home[1:] + "h"
			seq_away = seq_away[1:] + "b"
		else:
			seq_home = seq_home[1:] + "i"
			seq_away = seq_away[1:] + "a"

	elif len_home != window and len_away == window:
		if win == 1:
			seq_home += "h"
			seq_away = seq_away[1:] + "b"
		else:
			seq_home += "i"
			seq_away = seq_away[1:] + "a"

	elif len_home == window and len_away != window:
		if win == 1:
			seq_home = seq_home[1:] + "h"
			seq_away += "b"
		else:
			seq_home = seq_home[1:] + "i"
			seq_away += "a"

	else:
		if win == 1:
			seq_home += "h"
			seq_away += "b"
		else:
			seq_home += "i"
			seq_away += "a"

	# Renvoyer les séquences mises à jour
	return seq_home, seq_away

def calculate_streak_score(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calcule les scores de streak pour chaque équipe dans un DataFrame de données sportives.

	Args:
		data (pd.DataFrame): Le DataFrame contenant les données sportives, y compris les résultats des matchs.

	Returns:
		pd.DataFrame: Le DataFrame d'origine avec les colonnes supplémentaires contenant les score de streaks en continu pour chaque équipe.
	"""
	# Trier le dataframe par date
	data = data.sort_values(by="date").reset_index(drop=True)
	
	# Extraction des noms d'équipes uniques du DataFrame
	teams = np.unique(data["home"])

	# Initialisation du dictionnaire pour suivre les séquences de victoires/défaites par équipe
	streaks: Dict[str, str] = {team: "" for team in teams}
	
	# Initialiser la fenètre
	window = 5

	# Initialisation de la saison actuelle
	current_season = data.at[0, "season"]
	
	for index, row in data.iterrows():
		# Récupérer les infos
		home, away, season, win = row["home"], row["away"], row["season"], row["win"]

		# Si la saison change, réinitialiser les séquences de victoires/défaites
		if season != current_season:
			streaks = {team: "" for team in teams}
			current_season = season

		# Calculer les scores de streak en fonction de la séquence actuelle
		data.at[index, "streak_h"] = compute_streak_score(seq=streaks[home])
		data.at[index, "streak_a"] = compute_streak_score(seq=streaks[away])

		# Mettre à jour la séquence
		streaks[home], streaks[away] = update_streaks(seq_home=streaks[home], seq_away=streaks[away], win=win, window=window)

	# Renvoyer le DataFrame mis à jour
	return data


def calculate_LSR(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calcule le classement de la saison précédente (Last Season Ranking, LSR) pour des données sportives.

	Args:
		data (pd.DataFrame): Le DataFrame contenant les données des matchs sportifs.
	
	Returns:
		pd.DataFrame: Le DataFrame mis à jour avec les colonnes de classement de la saison précédente.
	"""
	# Trier le dataframe selon les dates
	data = data.sort_values(by=["date"]).reset_index(drop=True)

	# Classement initial à la fin de la saison 2002-2003
	LastSeasonRanking: Dict[str, int] = {
		"DAL": 30, "SAS": 29, "SAC": 28,
		"MIN": 27, "DET": 26, "POR": 25,
		"LAL": 24, "BKN": 23, "IND": 22,
		"PHI": 21, "UTA": 20, "NOP": 19,
		"PHX": 18, "BOS": 17, "HOU": 16,
		"MIL": 15, "ORL": 14, "OKC": 13,
		"GSW": 12, "WAS": 11, "NYK": 10,
		"ATL": 9, "CHI": 8, "MEM": 7,
		"LAC": 6, "MIA": 5, "TOR": 4,
		"DEN": 3, "CLE": 2, "CHA": 1
	}

	# Initialisation des dictionnaires pour suivre les résultats de la saison en cours
	CurrentSeasonOutcomes: Dict[str, int] = {team: 0 for team in data["home"].unique()}

	# Récupération de la saison en cours à partir de la première ligne
	current_season = data.at[0, "season"]

	for index, row in data.iterrows():
		# Récupérer les infos nécessaires
		home, away, season, win = row["home"], row["away"], row["season"], row["win"]

		# Vérification si la saison a changé
		if current_season != season:
			# Classement des équipes en fonction du nombre de victoires
			CurrentSeasonOutcomes = [(team, n_victoires) for team, n_victoires in CurrentSeasonOutcomes.items()]
			CurrentSeasonOutcomes = sorted(CurrentSeasonOutcomes, key=lambda item: item[1], reverse=False)
			CurrentSeasonOutcomes = [team for team, _ in CurrentSeasonOutcomes]
			
			# Mise à jour du classement de la saison précédente
			LastSeasonRanking = {team: i for i, team in enumerate(iterable=CurrentSeasonOutcomes, start=1)}
			CurrentSeasonOutcomes = {team: 0 for team in CurrentSeasonOutcomes}

			# Mettre à jour la saison
			current_season = season

		# Attribution du classement de la saison précédente aux équipes à domicile et à l'extérieur
		data.at[index, "lsr_h"] = LastSeasonRanking[home]
		data.at[index, "lsr_a"] = LastSeasonRanking[away]

		# Mise à jour des résultats de la saison en cours en fonction des victoires
		if win == 1:
			CurrentSeasonOutcomes[home] += 1
		else:
			CurrentSeasonOutcomes[away] += 1

	return data


def make_dataset() -> pd.DataFrame:
	"""
	Crée un DataFrame de jeu de données en combinant les saisons NBA.

	Returns:
		pd.DataFrame: Un DataFrame contenant les données combinées de plusieurs saisons NBA.
	"""
	# Création d'un DataFrame vide pour le jeu de données final
	dataset = pd.DataFrame()

	# Itération à travers les saisons spécifiées
	for season in SEASONS:
		# Appel de la fonction make_seasonDF pour chaque saison et concaténation des résultats
		dataset = pd.concat(objs=[dataset, make_seasonDF(season=season)], axis=0)

	# Réinitialisation des index
	dataset = dataset.reset_index(drop=True)

	# Mis à jour de nom d'équipes
	for index, row in dataset.iterrows():
		home, away = row["home"], row["away"]
		if home == "SEA":
			dataset.at[index, "home"] = "OKC"
		if away == "SEA":
			dataset.at[index, "away"] = "OKC"
		if home == "NJN":
			dataset.at[index, "home"] = "BKN"
		if away == "NJN":
			dataset.at[index, "away"] = "BKN"
		if home in ["NOH", "NOK"]:
			dataset.at[index, "home"] = "NOP"
		if away in ["NOH", "NOK"]:
			dataset.at[index, "away"] = "NOP"

	# Calcul de l'elo
	dataset = calculate_elos(data=dataset)
	
	# Calcul des winrates durant chaque saison
	dataset = calculate_laplace_proba(data=dataset)

	# Calcul dudes plus_minus
	dataset = calculate_plus_minus(data=dataset)

	# Calcul des séries de wins
	dataset = calculate_streak_score(data=dataset)

	# Calcul les ranking de la saison précédente
	dataset = calculate_LSR(data=dataset)
	
	# DataFrame final
	return dataset