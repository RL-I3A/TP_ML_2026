import sys
import numpy as np
from load_embeddings import load_embeddings
from plus_proche import construire, chercher   # <- à changer en fonction de la méthode


def calculer(texte, emb):
    # "roi - homme + femme" -> ["roi", "-", "homme", "+", "femme"]
    morceaux = texte.split()

    if len(morceaux) % 2 == 0:
        raise ValueError("syntaxe : mot (+ ou -) mot (+ ou -) mot ... (avec des espaces)")

    x = None
    utilises = []
    for i, m in enumerate(morceaux):
        if i % 2 == 0:                      # position d'un mot
            if m in ("+", "-"):
                raise ValueError(f"'{m}' est à une place où un mot est attendu")
            if m not in emb:
                raise ValueError(f"le mot '{m}' n'existe pas dans les embeddings")
            utilises.append(m)
            if x is None:
                x = emb[m].copy()
            elif operateur == "+":
                x = x + emb[m]
            else:
                x = x - emb[m]
        else:                               # position d'un opérateur
            if m not in ("+", "-"):
                raise ValueError(f"'{m}' n'est pas un opérateur (+ ou -)")
            operateur = m

    return x, utilises


if __name__ == "__main__":
    emb = load_embeddings(sys.argv[1], 500000)
    mots = list(emb.keys())
    matrice = np.array([emb[m] for m in mots])
    index = construire(mots, matrice)
    print(f"{len(mots)} mots chargés. Ligne vide pour quitter.")

    while True:
        texte = input("> ").strip()
        if texte == "":
            break
        try:
            x, utilises = calculer(texte, emb)
        except ValueError as e:
            print("erreur :", e)
            continue

        for mot, d in chercher(index, x, utilises):
            print(f"  {mot}  (distance {d:.3f})")

# python3 analogie_console.py 43/model.txt