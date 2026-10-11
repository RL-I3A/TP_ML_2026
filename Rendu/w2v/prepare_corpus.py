import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--max_lines", type=int, default=None, help="ne garder que les N premieres lignes (pour reduire la taille)")
args = parser.parse_args()

with open(args.input, encoding="utf-8") as f:
    lines = f.readlines()

if args.max_lines:
    lines = lines[:args.max_lines]


def recoller_bpe(tokens):
    # un token qui commence par # est un morceau du mot precedent
    mots = []
    for tok in tokens:
        if tok.startswith("#") and mots:
            mots[-1] += tok[1:]
        else:
            mots.append(tok)
    return mots


with open(args.output, "w", encoding="utf-8") as f:
    for line in lines:
        line = line.replace("<s>", "").replace("</s>", "")
        tokens = line.split()
        mots = recoller_bpe(tokens)
        if mots:
            f.write(" ".join(mots) + "\n")