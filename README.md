# nba_predictions_v2_fun

Ce projet vise à créer un modèle de machine learning pour prédire les résultats des matchs NBA. Le modèle est construit en utilisant des données des saisons régulières de la NBA depuis la saison 2003-2004.

## Structure du Projet

- **config.py:** Ce fichier contient des variables globales utilisées dans le projet.

- **PlotEval.py:** Ce fichier contient des fonctions pour le tracé de graphiques et l'évaluation du modèle.

- **CustomSplitting.py:** Ce fichier contient un splitter personnalisé nécessaire pour le projet.

- **MakeDataset.py:** Ce fichier permet la création du dataset avant le feature engineering.

- **FeaturesEngineering.py:** Ce fichier contient le code pour le feature engineering du dataset final.

- **build_datas.ipynb:** Ce notebook permet la création des datasets.

- **build_classifier.ipynb:** Ce notebook permet la création du classifieur de prédictions.

- **final_eval.ipynb:** Ce notebook permet la création de la fonction de correction du modèle.

- **dataJson/:** Ce dossier contient les données des saisons régulières NBA depuis la saison 2003-2004.

- **requirements.txt:** Ce fichier spécifie les dépendances nécessaires pour exécuter le projet.

## Utilisation

1. Clonez le projet depuis le lien GitHub :
    ```bash
    git clone https://github.com/RAVAO-Ravo/nba_prediction_v2_fun.git
    ```

2. Installez les dépendances en utilisant la commande suivante :
    ```bash
    pip3 install -r requirements.txt
    ```

3. Exécutez les notebooks `build_datas.ipynb`, `build_classifier.ipynb`, et `final_eval.ipynb` dans cet ordre pour construire, entraîner et évaluer le modèle.

## Licence

Ce projet est sous licence Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Vous êtes libre de :

- Partager : copier et redistribuer le matériel sous quelque support que ce soit ou sous n'importe quel format.
- Adapter : remixer, transformer et créer à partir du matériel.

Selon les conditions suivantes :

- Attribution : Vous devez donner le crédit approprié, fournir un lien vers la licence et indiquer si des modifications ont été apportées. Vous devez le faire de la manière suggérée par l'auteur, mais pas d'une manière qui suggère qu'il vous soutient ou soutient votre utilisation du matériel.

- Utilisation non commerciale : Vous ne pouvez pas utiliser le matériel à des fins commerciales.

[![Logo CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

[En savoir plus sur la licence CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)