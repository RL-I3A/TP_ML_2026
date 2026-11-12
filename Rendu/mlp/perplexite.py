import math
import torch

from model import LanguageModel
from train import make_loader, validate, Vocab, MLP_DIR


def main():
    model_path = MLP_DIR / "saved models/3.language_model_with_provided_embeddings.pt"
    test_file = MLP_DIR / "data/test_12k.bpe"
    vocab_file = MLP_DIR / "data/bpe_embeddings/bpe_tokenizer_vocab_12k.txt"

    device = "mps"

    #load saved model
    saved_model = torch.load(model_path, map_location=device, weights_only=True)
    config = saved_model["config"]

    #create model with same parameter
    model = LanguageModel(**config)
    model.load_state_dict(saved_model["model_state_dict"])
    model.to(device)

    vocab = Vocab(vocab_filename=str(vocab_file), emb_dim=config["embedding_dim"])
    test_loader = make_loader(test_file, vocab, k=config["k"], batch_size=256, shuffle=False)

    loss_function = torch.nn.CrossEntropyLoss()
    test_loss = validate(model, test_loader, loss_function, device)

    #calculate perplexity
    perplexity = math.exp(test_loss)
    log_perplexity_bits = test_loss / math.log(2)

    print(f"Test Loss: {test_loss}")
    print(f"Log-perplexity (bits): {log_perplexity_bits}")
    print(f"Perplexity: {perplexity}")


if __name__ == "__main__":
    main()
