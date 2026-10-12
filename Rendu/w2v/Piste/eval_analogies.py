import sys
import time
import numpy as np
from load_embeddings import load_embeddings
from ivf import construire, chercher   # à changer pour l'autre méthode


def trouver(mot, emb):
    # essaie le mot tel quel, puis en minuscules (France -> france)
    if mot in emb:
        return mot
    if mot.lower() in emb:
        return mot.lower()
    return None


if __name__ == "__main__":
    fichier_emb = sys.argv[1]
    fichier_analogies = sys.argv[2]
    max_mots = int(sys.argv[3]) if len(sys.argv) > 3 else None

    emb = load_embeddings(fichier_emb, max_mots)
    mots = list(emb.keys())
    matrice = np.array([emb[m] for m in mots])
    index = construire(mots, matrice)

    ok1 = 0
    ok5 = 0
    total = 0
    ignores = 0
    debut = time.time()

    with open(fichier_analogies, encoding="utf-8") as f:
        for num, ligne in enumerate(f, 1):
            q = ligne.split()
            if len(q) != 4:
                continue

            a = trouver(q[0], emb)
            b = trouver(q[1], emb)
            c = trouver(q[2], emb)
            d = trouver(q[3], emb)
            if a is None or b is None or c is None or d is None:
                ignores += 1
                print(f"{num:3d} IGNOREE  {ligne.strip()}")
                continue

            x = emb[b] - emb[a] + emb[c]
            resultats = chercher(index, x, [a, b, c], 5)
            predits = [mot for mot, dist in resultats]

            total += 1
            if predits[0] == d:
                ok1 += 1
            if d in predits:
                ok5 += 1

            statut = "OK " if predits[0] == d else "ERR"
            print(f"{num:3d} {statut}  attendu={d}  trouves={predits}")

    duree = time.time() - debut
    print()
    print(f"{total} évaluées, {ignores} ignorées (mot absent)")
    if total > 0:
        print(f"top-1 : {ok1}/{total} = {ok1/total:.4f}")
        print(f"top-5 : {ok5}/{total} = {ok5/total:.4f}")
    print(f"temps de recherche : {duree:.2f} s")