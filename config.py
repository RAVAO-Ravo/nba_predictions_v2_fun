#!/bin/python3
#-*- coding:utf-8 -*-

# Valeurs de paramètres
test_size: float = 0.30
class_weight: str = "balanced"
max_iter: int = 1000
random_state: int = 42
n_jobs: int = -1

# Chemins vers des fichiers
data_file: str = "./datas/dataset.csv"
x_train_file: str = "./datas/x_train.csv"
y_train_file: str = "./datas/y_train.csv"
x_test_file: str = "./datas/x_test.csv"
y_test_file: str = "./datas/y_test.csv"
classifier_file: str = "./classifiers/classifier.pkl"
proba_confident_file: str = "./classifiers/proba_confident.pkl"