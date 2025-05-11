
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("V2_1.2.0 – Entraînement LSTM PyTorch")

# 1. Chargement des données prétraitées
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# 2. Charger vocabulaire et label encoder
stoi = joblib.load('vocab_stoi.pkl')
le   = joblib.load('label_encoder.pkl')

# 3. Conversion en tensors et DataLoader
X_tr = torch.tensor(X_train, dtype=torch.long)
y_tr = torch.tensor(y_train, dtype=torch.long)
train_ds = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 4. Définition du modèle LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.out   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(self.embed(x))
        x = self.drop(x[:, -1, :])  # last output
        return self.out(x)

# 5. Hyperparamètres
vocab_size  = len(stoi) + 1
embed_dim   = 128
hidden_dim  = 64
num_classes = len(le.classes_)

model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. Entraînement
epochs = 15
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{epochs} – Loss: {total_loss/len(train_loader):.4f}")

# 7. Sauvegarde du modèle
torch.save(model.state_dict(), 'lstm_model.pth')
print("✅ Entraînement terminé, modèle enregistré sous 'lstm_model.pth'.")

import numpy as np
import joblib
import torch
import torch.nn as nn
import pandas as pd

print("V2_1.2.0 – Prédiction LSTM PyTorch")

# 1. Chargement des données et objets
X_test = np.load('X_test.npy')
stoi   = joblib.load('vocab_stoi.pkl')
le     = joblib.load('label_encoder.pkl')

# 2. Définition du modèle (doit correspondre à l'architecture du training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.out   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(self.embed(x))
        x = self.drop(x[:, -1, :])
        return self.out(x)

# 3. Initialiser et charger les poids
vocab_size  = len(stoi) + 1
embed_dim   = 128
hidden_dim  = 64
num_classes = len(le.classes_)

model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

# 4. Convertir X_test en tensor
X_te = torch.tensor(X_test, dtype=torch.long)

# 5. Prédiction
with torch.no_grad():
    logits = model(X_te)
    preds  = torch.argmax(logits, dim=1).cpu().numpy()
    cats   = le.inverse_transform(preds)

# 6. Génération du fichier de soumission
test_ids = pd.read_json('test_mini.json', orient='records')['Id']
submission = pd.DataFrame({'Id': test_ids, 'Category': cats})
submission.to_csv('submission2_3.csv', index=False)
print("✅ Prédiction terminée, 'submission.csv' généré.")
