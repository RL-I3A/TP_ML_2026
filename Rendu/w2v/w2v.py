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

    vocab = [] # Liste des mots (avec un certains taux d'apparition)
    for w in counts:
        if counts[w] >= minc:
            vocab.append(w)
    
    vocab.sort(key=lambda w: counts[w], reverse=True) # Trié par ordre de fréquence

    word2id = {} # Classement dans la liste : mot
    i = 0
    for w in vocab:
        word2id[w] = i
        i += 1

    vocab_set = set(vocab)
    filtered = []
    for s in sentences: # Pour chaque phrases
        s2 = [w for w in s if w in vocab_set]
        if len(s2) >= 2: # Si la phrase est trop petit on prend pas 
            filtered.append(s2)

    return word2id, vocab, filtered


def make_pairs(sentences_id, L):
    pairs = []
    for s in sentences_id: # Sentences_id = liste des phrases, où chaque mot est remplacé par son numéro
        n = len(s)
        for i in range(n):
            for j in range(max(0, i - L), min(n, i + L + 1)): # L = taille de demie fenêtre
                if i != j:
                    pairs.append((s[i], s[j]))
    return pairs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train(pairs, vocab_size, n, k, eta, epochs, seed=0):
    rng = np.random.default_rng(seed)
    M = (rng.random((vocab_size, n)) - 0.5) / n
    C = (rng.random((vocab_size, n)) - 0.5) / n

    for ep in range(epochs):
        random.shuffle(pairs)
        total_loss = 0
        for m, cpos in pairs:
            negs = rng.integers(0, vocab_size, size=k)

            v_m = M[m] # vecteur du mot cible
            v_cpos = C[cpos]
            v_negs = C[negs]

            sig_pos = sigmoid(np.dot(v_m, v_cpos))
            sig_negs = []
            for c_neg in v_negs:
                sig_negs.append(sigmoid(np.dot(v_m, c_neg)))

            total_loss += -np.log(sig_pos + 1e-10) - np.sum(np.log(1 - sig_negs + 1e-10))

            grad_cpos = (sig_pos - 1) * v_m
            grad_negs = sig_negs[:, None] * v_m[None, :]
            grad_m = (sig_pos - 1) * v_cpos + np.sum(sig_negs[:, None] * v_negs, axis=0)

            C[cpos] -= eta * grad_cpos
            C[negs] -= eta * grad_negs
            M[m] -= eta * grad_m

        print(f"epoch {ep+1}/{epochs} - loss = {total_loss/len(pairs):.4f}")

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
    args = parser.parse_args()

    sentences = read_corpus(args.corpus)
    word2id, vocab, sentences = build_vocab(sentences, args.minc)
    print(f"vocab size = {len(vocab)}")

    sentences_id = [[word2id[w] for w in s] for s in sentences]
    pairs = make_pairs(sentences_id, args.L)
    print(f"{len(pairs)} training pairs")

    M = train(pairs, len(vocab), args.n, args.k, args.eta, args.e)

    save_embeddings(args.output, vocab, M)
    print(f"embeddings written to {args.output}")


# python3 w2v.py corpus_bpe_gru.tok embeddings.txt --n 100 --L 2 --k 10 --eta 0.1 --e 5 --minc 5
# Tests
# sentences = read_corpus("corpus_bpe_gru.tok")
# print(sentences)