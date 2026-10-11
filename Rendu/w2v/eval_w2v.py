import argparse
import numpy as np


def load_embeddings(path):
    emb = {}
    with open(path, encoding="utf-8") as f:
        premiere_ligne = f.readline().split()
        dim = int(premiere_ligne[1])
        for line in f:
            parts = line.rstrip("\n").split(" ")
            word = parts[0]
            vec = np.array(parts[1:1 + dim], dtype=float)
            emb[word] = vec
    return emb


def load_triplets(path):
    triplets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            words = line.split()
            if len(words) == 3:
                triplets.append(tuple(words))
    return triplets


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("eval_file")
    args = parser.parse_args()

    emb = load_embeddings(args.embeddings)
    triplets = load_triplets(args.eval_file)

    ok = 0
    total = 0
    for m, mp, mn in triplets:
        if m not in emb or mp not in emb or mn not in emb:
            continue
        total += 1
        if cos(emb[m], emb[mp]) > cos(emb[m], emb[mn]):
            ok += 1

    print(f"{ok}/{total} triplets corrects ({total} sur {len(triplets)} evaluables)")
    print(f"taux de reussite : {ok/total:.4f}")


# python3 eval_w2v.py embeddings.txt Le_comte_de_Monte_Cristo.100.sim