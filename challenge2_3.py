
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

print("V2_1.1.0 – Entraînement + Prédiction PyTorch")

# 1. Chargement des artefacts prétraités
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test  = np.load('X_test.npy')
vectorizer = joblib.load('vectorizer.joblib')
le         = joblib.load('label_encoder.joblib')

# 2. Conversion en tensors PyTorch
X_tr = torch.tensor(X_train, dtype=torch.long)
y_tr = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)  # multiclass
X_te = torch.tensor(X_test, dtype=torch.long)

dataset = TensorDataset(X_tr, y_tr)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Modèle LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        _, (hn, _) = self.lstm(x)
        x = self.drop(hn[-1])
        return self.out(x)

vocab_size  = X_train.shape[1] + 1  # based on BOW padded dim
embed_dim   = 128
hidden_dim  = 64
num_classes = y_train.shape[1]
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Entraînement
epochs = 15
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}/{epochs} – Loss: {running_loss/len(loader):.4f}")

# 5. Prédiction sur X_test
model.eval()
with torch.no_grad():
    logits = model(X_te)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = le.inverse_transform(preds)

# 6. Génération de la soumission
print("Saving submission.csv...")
test_ids = pd.read_json('test_mini.json', orient='records')['Id']
submission = pd.DataFrame({
    'Id': test_ids,
    'Category': labels
})
submission.to_csv('submission2_3.csv', index=False)
print("✅ submission.csv généré.")
