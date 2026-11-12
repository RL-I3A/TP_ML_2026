## Données

Le corpus d'entraînement de word2vec est inclus dans le dépôt :

- `Rendu/w2v/data/train_500k.tok` : corpus brut (environ 500 000 tokens), partagé au sein de la promotion
- `Rendu/w2v/data/train_500k_propre.tok` : le même découpé en phrases et sans ponctuation, obtenu avec :

```
cd Rendu/w2v
python3 data/prepare_500k.py data/train_500k.tok data/train_500k_propre.tok
```

Non inclus (trop volumineux), à placer dans un dossier `Donnees/` à la racine du projet :

1. Décompresser l'archive `talia_donnees.tar` et placer le dossier `talia_donnees/` dans `Donnees/`
2. Récupérer le modèle 43 sur https://vectors.nlpl.eu/repository/ (utilisé par la piste 8.2) et placer le dossier `43/` dans `Donnees/`

```
TP_ML/
├── Donnees/
│   ├── talia_donnees/
│   └── 43/
└── Rendu/
    └── w2v/
        ├── data/
        ├── res/
        └── Piste/
            ├── Piste1/
            └── Piste2/
```
