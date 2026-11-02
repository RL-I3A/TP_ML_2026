import sys
import re

entree = sys.argv[1]
sortie = sys.argv[2]

with open(entree, encoding="utf-8") as f:
    texte = f.read()

# le fichier est sur une seule ligne : on le recoupe en phrases sur . ! ?
phrases = re.split(r" [.!?] ", texte)

with open(sortie, "w", encoding="utf-8") as f:
    for p in phrases:
        mots = []
        for t in p.split():
            if re.search(r"\w", t):     # on garde un token s'il contient au moins une lettre
                mots.append(t)
        if len(mots) >= 2:
            f.write(" ".join(mots) + "\n")