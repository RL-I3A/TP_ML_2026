# Lance plusieurs entraînements par configuration (une graine différente à chaque fois),
# évalue chacun avec eval_w2v.py, enregistre tout dans res/resultats.csv
# et affiche la moyenne et l'écart-type de chaque configuration.
#
# Utilisation (depuis le dossier Piste1) :
#   python3 experiences.py 10 base neg075
#   -> 10 graines pour la config "base" puis 10 pour "neg075"
#   python3 experiences.py resume
#   -> affiche le résumé de tout ce qui est dans res/resultats.csv

import sys
import subprocess
import time
import numpy as np

CORPUS = "../../data/train_500k_propre.tok"
EVAL = "../../data/Le_comte_de_Monte_Cristo.100.sim"
EVAL_SCRIPT = "../../eval_w2v.py"
SORTIE = "res/emb_tmp.emb"            # écrasé à chaque lancement (.emb est dans le .gitignore)
RESULTATS = "res/resultats.csv"

# une seule amélioration à la fois, puis tout ensemble
CONFIGS = {
    "base": [],
    "neg075": ["--neg", "unigram"],
    "sousech_1e-3": ["--t", "1e-3"],
    "sousech_1e-4": ["--t", "1e-4"],
    "sousech_1e-5": ["--t", "1e-5"],
    "decroissance": ["--decroissance"],
    "fenetre_alea": ["--fenetre_aleatoire"],
    "c_zero": ["--c_zero"],
    # à adapter : garder la meilleure valeur de t trouvée au-dessus
    "neg_decr": ["--neg", "unigram", "--decroissance"],
    "neg_decr_czero": ["--neg", "unigram", "--decroissance", "--c_zero"],
}


def lancer(config, graine):
    commande = [sys.executable, "w2v_ameliore.py", CORPUS, SORTIE, "--seed", str(graine)] + CONFIGS[config]
    subprocess.run(commande, check=True, stdout=subprocess.DEVNULL)

    sortie = subprocess.run([sys.executable, EVAL_SCRIPT, SORTIE, EVAL],
                            check=True, capture_output=True, text=True).stdout
    # dernière ligne : "taux de reussite : 0.7442"
    taux = float(sortie.strip().split("\n")[-1].split(":")[1])
    evaluables = sortie.split("(")[1].split(" ")[0]
    return taux, evaluables


def resume():
    scores = {}
    with open(RESULTATS, encoding="utf-8") as f:
        for ligne in f:
            config, graine, taux, evaluables, duree = ligne.strip().split(",")
            if config not in scores:
                scores[config] = []
            scores[config].append(float(taux))

    print(f"{'config':15s} {'nb':>3s} {'moyenne':>8s} {'écart-type':>10s}")
    for config in scores:
        s = np.array(scores[config])
        ecart = s.std(ddof=1) if len(s) > 1 else 0.0
        print(f"{config:15s} {len(s):3d} {s.mean():8.4f} {ecart:10.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "resume":
        resume()
        sys.exit()

    nb_graines = int(sys.argv[1])
    configs = sys.argv[2:]
    for config in configs:
        if config not in CONFIGS:
            print(f"config inconnue : {config}. Choix : {list(CONFIGS)}")
            sys.exit()

    for config in configs:
        for graine in range(nb_graines):
            debut = time.time()
            taux, evaluables = lancer(config, graine)
            duree = time.time() - debut
            print(f"{config:15s} graine {graine}  taux = {taux:.4f}  ({evaluables} évaluables, {duree:.0f} s)", flush=True)
            with open(RESULTATS, "a", encoding="utf-8") as f:
                f.write(f"{config},{graine},{taux},{evaluables},{duree:.0f}\n")

    print()
    resume()
