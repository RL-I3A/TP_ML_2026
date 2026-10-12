import numpy as np


def plus_proche_centre(matrice, centres):
    # pour chaque mot, numéro du centre le plus proche
    etiquettes = np.zeros(len(matrice), dtype=int)
    norme_c = np.sum(centres ** 2, axis=1)            # ||c||² pour chaque centre

    # par blocs pour éviter une matrice qui serait trop grosse
    for debut in range(0, len(matrice), 10000):
        bloc = matrice[debut:debut + 10000]
        # ||x - c||² = ||x||² - 2 x·c + ||c||²
        # ||x||² est le même pour tous les centres, on l'ignore pour comparer
        scores = norme_c[None, :] - 2 * (bloc @ centres.T)
        etiquettes[debut:debut + 10000] = np.argmin(scores, axis=1)
    return etiquettes


def kmeans(donnees, k, nb_iter, rng):
    # on part de k mots tirés au hasard comme centres
    centres = donnees[rng.choice(len(donnees), k, replace=False)].copy()

    for it in range(nb_iter):
        etiquettes = plus_proche_centre(donnees, centres)
        for j in range(k):
            membres = donnees[etiquettes == j]
            if len(membres) > 0:                      # un groupe vide garde son ancien centre
                centres[j] = membres.mean(axis=0)
        print(f"k-means : itération {it + 1}/{nb_iter}")
    return centres


def construire(mots, matrice, nb_groupes=None, nprobe=10, nb_iter=10):
    rng = np.random.default_rng(0)
    N = len(mots)
    if nb_groupes is None:
        nb_groupes = max(1, int(np.sqrt(N)))

    taille = min(N, 50000)
    echantillon = matrice[rng.choice(N, taille, replace=False)]
    centres = kmeans(echantillon, nb_groupes, nb_iter, rng)

    # chaque mot est rangé dans le groupe de son centre le plus proche
    etiquettes = plus_proche_centre(matrice, centres)
    groupes = []
    for j in range(nb_groupes):
        groupes.append(np.where(etiquettes == j)[0])  # numéros des mots du groupe j

    return (mots, matrice, centres, groupes, nprobe)


def chercher(index, x, exclus, topn=5):
    mots, matrice, centres, groupes, nprobe = index

    dist_centres = np.linalg.norm(centres - x, axis=1)

    groupes_choisis = np.argsort(dist_centres)[:nprobe]

    listes = []
    for g in groupes_choisis:
        listes.append(groupes[g])
    candidats = np.concatenate(listes)

    dist = np.linalg.norm(matrice[candidats] - x, axis=1)
    ordre = np.argsort(dist)

    resultats = []
    for i in ordre:
        mot = mots[candidats[i]]
        if mot not in exclus:
            resultats.append((mot, dist[i]))
            if len(resultats) == topn:
                break
    return resultats