# Piste 8.1 : amélioration des performances de w2v.py
#
# Améliorations reprises de Mikolov et al. (2013), "Distributed Representations of
# Words and Phrases and their Compositionality" (article) et du code officiel word2vec :
#   --neg unigram        négatifs tirés selon fréquence^0.75 (article, section 2.2)
#   --t 1e-4             sous-échantillonnage des mots fréquents (article, section 2.3)
#   --decroissance       taux d'apprentissage qui diminue linéairement (code officiel)
#   --fenetre_aleatoire  taille de fenêtre tirée entre 1 et L pour chaque mot (code officiel)
#   --c_zero             matrice C initialisée à 0 (code officiel)
# Sans aucune option, c'est exactement le modèle de base de w2v.py.

import argparse
import random
import numpy as np


def read_corpus(path):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append(tokens)
    return sentences


def build_vocab(sentences, minc):
    counts = {}
    for s in sentences:
        for w in s:
            if w in counts:
                counts[w] += 1
            else:
                counts[w] = 1

    vocab = []
    for w in counts:
        if counts[w] >= minc:
            vocab.append(w)
    vocab.sort(key=lambda w: counts[w], reverse=True)

    word2id = {}
    i = 0
    for w in vocab:
        word2id[w] = i
        i += 1

    # nombre d'occurrences de chaque mot du vocabulaire (même ordre que vocab)
    comptes = []
    for w in vocab:
        comptes.append(counts[w])
    comptes = np.array(comptes, dtype=float)

    vocab_set = set(vocab)
    filtered = []
    for s in sentences:
        s2 = [w for w in s if w in vocab_set]
        if len(s2) >= 2:
            filtered.append(s2)

    return word2id, vocab, comptes, filtered


def sous_echantillonner(sentences_id, freq, t, rng):
    # Article, équation (5) : chaque occurrence d'un mot w est supprimée
    # avec la probabilité 1 - sqrt(t / f(w)), f(w) = fréquence relative de w
    p_garder = np.minimum(1.0, np.sqrt(t / freq))
    nouvelles = []
    for s in sentences_id:
        tirages = rng.random(len(s))
        s2 = []
        for i in range(len(s)):
            if tirages[i] < p_garder[s[i]]:
                s2.append(s[i])
        if len(s2) >= 2:
            nouvelles.append(s2)
    return nouvelles


def make_pairs(sentences_id, L, fenetre_aleatoire=False, rng=None):
    pairs = []
    for s in sentences_id:
        n = len(s)
        for i in range(n):
            if fenetre_aleatoire:
                l = rng.integers(1, L + 1)   # code officiel : fenêtre entre 1 et L
            else:
                l = L
            for j in range(max(0, i - l), min(n, i + l + 1)):
                if i != j:
                    pairs.append((s[i], s[j]))
    return pairs


def table_negatifs(comptes, puissance, taille=10_000_000):
    # Article : les négatifs sont tirés selon P(w) = U(w)^(3/4) / Z
    # On remplit une grande table où chaque mot apparaît proportionnellement à P(w),
    # puis on tire une case au hasard (même idée que le code officiel)
    p = comptes ** puissance
    p = p / p.sum()
    nb_cases = np.maximum(1, np.round(p * taille)).astype(int)
    return np.repeat(np.arange(len(p)), nb_cases)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train(sentences_id, comptes, args):
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    V = len(comptes)
    n = args.n

    M = (rng.random((V, n)) - 0.5) / n
    if args.c_zero:
        C = np.zeros((V, n))
    else:
        C = (rng.random((V, n)) - 0.5) / n

    table = None
    if args.neg == "unigram":
        table = table_negatifs(comptes, args.puissance)
    freq = comptes / comptes.sum()

    # si la construction des paires ne dépend pas du hasard, on la fait une seule fois
    paires_fixes = None
    if args.t == 0 and not args.fenetre_aleatoire:
        paires_fixes = make_pairs(sentences_id, args.L)

    for ep in range(args.e):
        if paires_fixes is not None:
            pairs = paires_fixes
        else:
            # sous-échantillonnage et fenêtre aléatoire : nouveau tirage à chaque époque
            phrases = sentences_id
            if args.t > 0:
                phrases = sous_echantillonner(sentences_id, freq, args.t, rng)
            pairs = make_pairs(phrases, args.L, args.fenetre_aleatoire, rng)

        random.shuffle(pairs)
        total_loss = 0
        nb = len(pairs)
        eta = args.eta

        for idx in range(nb):
            m, cpos = pairs[idx]

            if args.decroissance:
                avancement = (ep + idx / nb) / args.e          # de 0 à 1 sur tout l'entraînement
                eta = args.eta * max(1 - avancement, 0.0001)

            if table is not None:
                negs = table[rng.integers(0, len(table), size=args.k)]
            else:
                negs = rng.integers(0, V, size=args.k)

            v_m = M[m]
            v_cpos = C[cpos]
            v_negs = C[negs]

            sig_pos = sigmoid(np.dot(v_m, v_cpos))
            sig_negs = sigmoid(v_negs @ v_m)     # même calcul que la boucle de w2v.py, en plus rapide

            total_loss += -np.log(sig_pos + 1e-10) - np.sum(np.log(1 - sig_negs + 1e-10))

            grad_cpos = (sig_pos - 1) * v_m
            grad_negs = sig_negs[:, None] * v_m[None, :]
            grad_m = (sig_pos - 1) * v_cpos + np.sum(sig_negs[:, None] * v_negs, axis=0)

            C[cpos] -= eta * grad_cpos
            C[negs] -= eta * grad_negs
            M[m] -= eta * grad_m

        print(f"epoch {ep+1}/{args.e} - {nb} paires - loss = {total_loss/nb:.4f}")

    return M


def save_embeddings(path, vocab, M):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {M.shape[1]}\n")
        for i, w in enumerate(vocab):
            vec = " ".join(f"{x:.6f}" for x in M[i])
            f.write(f"{w} {vec}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("output")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--e", type=int, default=5)
    parser.add_argument("--minc", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    # améliorations (désactivées par défaut)
    parser.add_argument("--neg", default="uniforme")              # uniforme ou unigram
    parser.add_argument("--puissance", type=float, default=0.75)  # exposant pour --neg unigram
    parser.add_argument("--t", type=float, default=0)             # 0 = pas de sous-échantillonnage
    parser.add_argument("--decroissance", action="store_true")
    parser.add_argument("--fenetre_aleatoire", action="store_true")
    parser.add_argument("--c_zero", action="store_true")
    args = parser.parse_args()

    sentences = read_corpus(args.corpus)
    word2id, vocab, comptes, sentences = build_vocab(sentences, args.minc)
    print(f"vocab size = {len(vocab)}")

    sentences_id = [[word2id[w] for w in s] for s in sentences]

    M = train(sentences_id, comptes, args)

    save_embeddings(args.output, vocab, M)
    print(f"embeddings written to {args.output}")
