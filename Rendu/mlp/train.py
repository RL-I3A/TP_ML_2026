from pathlib import Path
import sys
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import LanguageModel

# Resolve paths relative to this file, regardless of the working directory.
MLP_DIR = Path(__file__).resolve().parent
ROOT = MLP_DIR.parent.parent
sys.path.insert(0, str(ROOT / "Sujets"))
from Vocab import Vocab


def prepare_data(text_file, vocab, k):
    sequences = []
    with open(text_file, "r", encoding="utf-8") as file:
        # Read the file line by line
        for line in file: 
            # Split the line into tokens
            tokens = line.split()

            # Convert each token to its ID.
            ids = []
            for token in tokens:
                token_id = vocab.get_word_index2(token)
                ids.append(token_id)
    
            # Create windows with k context tokens and one target token
            number_of_lists = len(ids) - k

            for start in range(number_of_lists):
                window = []

                for offset in range(k + 1):
                    position = start + offset
                    token_id = ids[position]
                    window.append(token_id)

                sequences.append(window)
    return sequences

def make_loader(text_file, vocab, k, batch_size, shuffle):
    sequences = prepare_data(text_file, vocab, k)

    data = torch.tensor(sequences)
    contexts = data[:, :-1]
    targets = data[:, -1]

    dataset = TensorDataset(contexts, targets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def validate(model, val_loader, loss_function, device="cpu"):
    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device)
            target = target.to(device)

            output = model(context)
            loss = loss_function(output, target)

            batch_size = target.size(0)
            val_loss_sum += loss.item() * batch_size
            val_count += batch_size

    return val_loss_sum / val_count


def train(model, train_loader, val_loader, epochs, learning_rate, patience, device, saved_model_path, vocab):
    # Save one row per completed epoch beside the model file.
    csv_path = Path(saved_model_path).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(["epoch", "train_loss", "val_loss"])

    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    #for early stop calculations
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss_sum = 0.0
        train_count = 0

        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(context)

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            batch_size = target.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        val_loss = validate(model, val_loader, loss_function, device)
        print(f"Epoch {epoch + 1}: train={train_loss}, val={val_loss}")

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([epoch + 1, train_loss, val_loss])

        #Early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            #Save best model with torch.save
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": model.embedding.num_embeddings,
                    "embedding_dim": model.embedding.embedding_dim,
                    "k": model.hidden.in_features // model.embedding.embedding_dim,
                    "hidden_dim": model.hidden.out_features,
                    "train_embeddings": model.embedding.weight.requires_grad,
                },
                "word_to_index": vocab.dico_voca,
            }, saved_model_path)
            print(f"Best model saved: {saved_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early Stopping: no improvement after", epochs_without_improvement, "epochs")
            break

def load_provided_embeddings(model, vocab, embedding_file):
    provided = Vocab(emb_filename=str(embedding_file))
    with torch.no_grad():
        for token, token_id in vocab.dico_voca.items():
            if token in provided.dico_voca:
                model.embedding.weight[token_id] = provided.matrice[
                    provided.dico_voca[token]
                ]

def main():
    k = 3
    embedding_dim = 32
    hidden_dim = 256
    batch_size = 256

    train_file = MLP_DIR / "data/train_12k.bpe"
    val_file = MLP_DIR / "data/val_12k.bpe"
    vocab_file = MLP_DIR / "data/bpe_embeddings/bpe_tokenizer_vocab_12k.txt"
    embedding_file = MLP_DIR / "data/bpe_embeddings/bpe_embeddings_12k_32.txt"
    model_name = "3.language_model_with_provided_embeddings"

    saved_models_dir = MLP_DIR / "saved models"
    saved_models_dir.mkdir(exist_ok=True)
    saved_model_path = saved_models_dir / f"{model_name}.pt"

    # Use the same vocabulary for training and validation
    vocab = Vocab(vocab_filename=str(vocab_file), emb_dim=embedding_dim)

    train_loader = make_loader(train_file, vocab, k, batch_size, shuffle=True)
    val_loader = make_loader(val_file, vocab, k, batch_size, shuffle=False)

    print("Training examples:",len(train_loader.dataset), ", Validation examples:", len(val_loader.dataset))

    model = LanguageModel(vocab.vocab_size, embedding_dim, k, hidden_dim, train_embeddings=False)

    #use provided embeddings
    load_provided_embeddings(model, vocab, embedding_file)
    
    train(
        model, 
        train_loader, 
        val_loader, 
        epochs=50, 
        learning_rate=0.001, 
        patience=5, 
        device="mps", 
        saved_model_path=saved_model_path, 
        vocab=vocab)

if __name__ == "__main__":
    main()
