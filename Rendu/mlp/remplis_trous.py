import torch
from model import LanguageModel
from train import MLP_DIR


def fill_blanks(model_file, text_file, candidates_file, device="cpu"):
    saved_model = torch.load(model_file, map_location="cpu", weights_only=True)
    model = LanguageModel(**saved_model["config"]).to(device)
    model.load_state_dict(saved_model["model_state_dict"])
    model.eval()
    vocab = saved_model["word_to_index"]
    k = saved_model["config"]["k"]
    candidates = candidates_file.read_text(encoding="utf-8").split()
    
    scores = []
    with torch.no_grad(), text_file.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            fields = line.split()

            position = int(fields[0])
            tokens = fields[1:]
            answer = tokens[position]

            choices = candidates.copy()
            if answer not in choices:
                choices[-1] = answer

            ids = [vocab[token] for token in tokens]
            
            positions = range(position, min(len(ids), position + k + 1))
            contexts, targets = [], []
            for candidate in choices:
                completed = ids.copy()
                completed[position] = vocab[candidate]
                for i in positions:
                    contexts.append(completed[i-k:i])
                    targets.append(completed[i])

            contexts = torch.tensor(contexts, device=device)
            targets = torch.tensor(targets, device=device)
            log_probs = model(contexts).log_softmax(dim=-1)
            token_scores = log_probs.gather(1, targets[:, None]).squeeze(1)
            sentence_scores = token_scores.reshape(100, len(positions)).sum(dim=1)
            ranking = sentence_scores.argsort(descending=True).tolist()
            rank = ranking.index(choices.index(answer)) + 1
            normalized_rank = (100 - rank + 1) / 100
            scores.append(normalized_rank)

    mean_rank = sum(scores) / len(scores)
    print(f"Mean normalized rank: {mean_rank}")
    return mean_rank


def main():
    model_file = MLP_DIR / "Saved Models/2.language_model_baseline.pt"
    device = "mps"

    # Evaluate all three texts with each of the ten candidate sets.
    for text_number in range(1, 4):
        text_file = (
            MLP_DIR / "data/corpus/balises_mlp/mask_files_100_sentences"
            / f"file_{text_number}.mask"
        )
        for candidate_number in range(1, 11):
            candidates_file = (
                MLP_DIR / "data/tokens_candidats_textes_a_trou_12k"
                / f"ensemble_{candidate_number}.txt"
            )
            print("\nText:", text_number, "Candidate set:", candidate_number)
            fill_blanks(model_file, text_file, candidates_file, device)


if __name__ == "__main__":
    main()
