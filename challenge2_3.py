import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

print("V2_1.2.1 – Entraînement + Prédiction LSTM PyTorch")

# 1. Charger artefacts
X_train = np.load('X_train_seq.npy')
y_train = np.load('y_train.npy')
X_test  = np.load('X_test_seq.npy')
stoi     = joblib.load('vocab_stoi.pkl')
le       = joblib.load('label_encoder.pkl')

# 2. Convertir en tensors
X_tr = torch.tensor(X_train, dtype=torch.long)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.long)

train_ds = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 3. Définir le modèle LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb, _ = self.lstm(self.embedding(x))
        last = emb[:, -1, :]
        return self.fc(self.dropout(last))

vocab_size  = len(stoi)
embed_dim   = 128
hidden_dim  = 64
num_classes = len(le.classes_)

model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=stoi['<PAD>'])

# 4. Configuration de l’entraînement
# 4.1 Calcul manuel des poids de classes
counts = np.bincount(y_train, minlength=num_classes)
total  = y_train.shape[0]
# éviter division par zéro au cas où une classe n'apparaît pas
counts = np.where(counts == 0, 1, counts)
class_weights = torch.tensor((total / counts), dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs    = 15

# 5. Entraînement
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total
    print(f"Epoch {epoch}/{epochs} — loss: {train_loss:.4f} — acc: {train_acc:.4f}")

# 6. Prédiction sur le test
model.eval()
with torch.no_grad():
    logits = model(X_te)
    preds  = torch.argmax(logits, dim=1).cpu().numpy()
    cats   = le.inverse_transform(preds)

# 7. Écriture de la soumission
print("Saving submission2_3.csv...")
test_ids   = pd.read_json('test_mini.json', orient='records')['Id']
submission = pd.DataFrame({'Id': test_ids, 'Category': cats})
submission.to_csv('submission2_3.csv', index=True)
print("✅ submission2_3.csv généré.")
