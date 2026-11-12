from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, k, hidden_dim, train_embeddings=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad_(train_embeddings)

        #hidden layer
        self.hidden = nn.Linear(k*embedding_dim,hidden_dim)
        self.relu = nn.ReLU()

        #output_layer
        self.output = nn.Linear (hidden_dim,vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, context):
        embeddings = self.embedding(context)
        x = embeddings.flatten(start_dim=1)

        h = self.hidden(x)
        h = self.relu(h)

        logits = self.output(h)

        return logits
