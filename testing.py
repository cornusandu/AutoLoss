import json
import json
import torch
import torch.nn as nn
from autoloss import autoloss
import datasets
from transformers import AutoTokenizer
import colorama

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Definirea modelului pentru clasificarea textului toxic
class ToxicTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=3):
        super(ToxicTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # Obținerea ieșirii din LSTM; se utilizează ultima stare ascunsă pentru predicție
        output, (hidden, _) = self.lstm(embedded)
        logits = self.fc(hidden[-1])
        return logits


# Setări ipotetice pentru vocabular, dimensiunea embedding-ului și dimensiunea stratului ascuns
VOCAB_SIZE = tokenizer.vocab_size  # Dimensiunea vocabularului (exemplu)
EMBED_DIM = 128
HIDDEN_DIM = 64

# Instanțierea modelului
model = ToxicTextClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

loss_fn = nn.CrossEntropyLoss()

with open('test_data.json', 'r') as f:
    toxicity_dataset = json.load(f)

EPOCHS = 100

optimizer = autoloss.AutoLoss(model, loss_fn, patience=25)

loss_avg = 0
iterations = 1

for epoch in range(EPOCHS):
    _loss_avg = 0
    _losses = []
    for data in toxicity_dataset[:3]:
        iterations += 1
        tokens = tokenizer.encode(data['text'], add_special_tokens=False)
        score = round(data['toxicity'] * 2)
        loss = optimizer.step(torch.tensor(tokens, dtype=torch.int64).unsqueeze(0),
                              torch.tensor([score], dtype=torch.long))
        _losses.append(loss)

        loss_avg = (loss_avg * (iterations - 1) + loss) / iterations
    _loss_avg = sum(_losses) / len(_losses)

    if _loss_avg < loss_avg:
        print(f"{colorama.Fore.GREEN}Epoch: {epoch + 1}, Loss: {_loss_avg}{colorama.Fore.RESET}")
    else:
        print(f"{colorama.Fore.RED}Epoch: {epoch + 1}, Loss: {_loss_avg}{colorama.Fore.RESET}")
