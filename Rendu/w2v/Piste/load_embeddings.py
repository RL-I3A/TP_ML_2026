import argparse
import numpy as np


def load_embeddings(path, max_mots=None):
    emb = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        n, dim = map(int, f.readline().split())
        for line in f:
            parts = line.split()
            if len(parts) <= dim:           # ligne cassées
                continue
            word = " ".join(parts[:-dim])   # tout sauf les dim derniers nombres
            vec = np.array(parts[-dim:], dtype=float)
            emb[word] = vec
            if max_mots and len(emb) >= max_mots:
                break
    return emb
