import sys
import time
import argparse
import numpy as np
from load_embeddings import load_embeddings


def trouver(mot, emb):
    # essaie le mot tel quel, puis en minuscules (France -> france)
    if mot in emb:
        return mot
    if mot.lower() in emb:
        return mot.lower()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("analogies")
    parser.add_argument("--methode", default="naive")      # naive, kd ou ivf
    parser.add_argument("--nprobe", type=int, default=10)  # utilisé seulement par ivf
    parser.add_argument("--max_mots", type=int, default=None)
    args = parser.parse_args()

    if args.methode == "naive":
        from plus_proche import construire, chercher
    elif args.methode == "kd":
        import kd
        from kd import construire, chercher
    elif args.methode == "ivf":
        from ivf import construire, chercher
    else:
        print("méthode inconnue (naive, kd ou ivf)")
        sys.exit()

    emb = load_embeddings(args.embeddings, args.max_mots)
    mots = list(emb.keys())
    matrice = np.array([emb[m] for m in mots])
    print(f"{len(mots)} mots chargés, méthode = {args.methode}")

    t0 = time.time()
    if args.methode == "ivf":
        index = construire(mots, matrice, nprobe=args.nprobe)
    else:
        index = construire(mots, matrice)
    print(f"temps de construction : {time.time() - t0:.2f} s")

    ok1 = 0
    ok5 = 0
    total = 0
    ignores = 0
    debut = time.time()

    with open(args.analogies, encoding="utf-8") as f:
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
        print(f"temps de recherche : {duree:.2f} s ({1000 * duree / total:.1f} ms par analogie)")
    if args.methode == "kd" and total > 0:
        print(f"mots examinés par requête : {100 * kd.stats['examines'] / total / len(mots):.1f} %")