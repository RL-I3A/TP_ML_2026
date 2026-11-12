import torch

from model import LanguageModel
from train import MLP_DIR


def generate(model_file, max_tokens=100, device="cpu"):
    saved_model = torch.load(model_file, map_location="cpu", weights_only=True)
    model = LanguageModel(**saved_model["config"]).to(device)
    model.load_state_dict(saved_model["model_state_dict"])
    model.eval()
    vocab = saved_model["word_to_index"]

    id_to_token = {}
    for token, index in vocab.items():
        id_to_token[index] = token
        
    k = saved_model["config"]["k"]
    context = []

    for i in range(k):
        context.append(vocab["<s>"])
    generated = []

    with torch.no_grad():
        for _ in range(max_tokens):
            inputs = torch.tensor([context], device=device)
            probabilities = model(inputs).softmax(dim=-1)
            
            # Sampling, rather than always choosing the most likely token.
            token_id = torch.multinomial(probabilities, 1).item()
            token = id_to_token[token_id]
            if token == "</s>":
                break
            generated.append(token)
            context = context[1:] + [token_id]

    print(" ".join(generated))


def main():
    model_file = MLP_DIR / "Saved Models/3.language_model_with_provided_embeddings.pt"
    max_tokens = 100
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generate(model_file, max_tokens, device)


if __name__ == "__main__":
    main()
