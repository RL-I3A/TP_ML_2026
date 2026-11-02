import numpy as np

stats = {"examines": 0}      # compteur : nombre de mots dont on a calculé la distance


def construire_noeud(matrice, idx, taille_feuille):
    # peu de mots restants : une feuille
    if len(idx) <= taille_feuille:
        return ("feuille", idx)

    points = matrice[idx]
    dim = np.argmax(np.var(points, axis=0))      # dimension où les mots sont le plus étalés
    ordre = np.argsort(points[:, dim])
    idx = idx[ordre]                             # mots triés selon cette dimension

    m = len(idx) // 2                            # position du médian
    valeur = matrice[idx[m], dim]                # valeur de coupe

    gauche = construire_noeud(matrice, idx[:m], taille_feuille)
    droite = construire_noeud(matrice, idx[m:], taille_feuille)
    return ("noeud", dim, valeur, gauche, droite)


def construire(mots, matrice, taille_feuille=20):
    racine = construire_noeud(matrice, np.arange(len(mots)), taille_feuille)
    return (mots, matrice, racine)


def chercher(index, x, exclus, topn=5):
    mots, matrice, racine = index

    # on cherche topn + len(exclus) voisins, car on va retirer les mots exclus ensuite
    k = topn + len(exclus)
    meilleurs = []                              

    def visiter(noeud):
        if noeud[0] == "feuille":
            idx = noeud[1]
            d2 = np.sum((matrice[idx] - x) ** 2, axis=1)
            stats["examines"] += len(idx)
            for i in range(len(idx)):
                meilleurs.append((d2[i], idx[i]))
            meilleurs.sort()
            del meilleurs[k:]                    
            return

        _, dim, valeur, gauche, droite = noeud
        ecart = x[dim] - valeur
        if ecart < 0:
            proche, loin = gauche, droite
        else:
            proche, loin = droite, gauche

        visiter(proche)
        if len(meilleurs) < k or ecart ** 2 < meilleurs[-1][0]:
            visiter(loin)

    visiter(racine)

    resultats = []
    for d2, i in meilleurs:
        if mots[i] not in exclus:
            resultats.append((mots[i], np.sqrt(d2)))
            if len(resultats) == topn:
                break
    return resultats