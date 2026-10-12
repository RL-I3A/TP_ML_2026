import numpy as np


def construire(mots, matrice):
    return (mots, matrice) # utile seulement pour l'uatre méthode


def chercher(index, x, exclus, topn=5):
    mots, matrice = index

    # distance euclidienne entre x et CHAQUE mot du vocabulaire
    dist = np.linalg.norm(matrice - x, axis=1)

    # indices des mots, du plus proche au plus loin
    ordre = np.argsort(dist)

    resultats = []
    for i in ordre:
        if mots[i] not in exclus:
            resultats.append((mots[i], dist[i]))
            if len(resultats) == topn:
                break
    return resultats