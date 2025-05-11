import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

print("V2_1.3.1 – Entraînement + Prédiction PyTorch corrigé")

# 1. Charger les artefacts prétraités
X_train = np.load('X_train.npy')        # représentation bag-of-words pad/trunc
y_train_onehot = np.load('y_train.npy') # one-hot
X_test = np.load('X_test.npy')          # représentation bag-of-words pad/trunc

vectorizer = joblib.load('vectorizer.joblib')     # transformateur BOW
le = joblib.load('label_encoder.joblib')           # encodeur de labels

# 2. Préparer les labels pour CrossEntropyLoss
# Convertir one-hot en indices de classes
y_train = np.argmax(y_train_onehot, axis=1)

# 3. Conversion en tensors PyTorch
dtype = torch.float32
X_tr = torch.tensor(X_train, dtype=dtype)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test, dtype=dtype)

dataset = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Définition du modèle MLP pour BOW
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]  # longueur du vecteur BOW
hidden_dim = 64
num_classes = y_train_onehot.shape[1]
model = SimpleMLP(input_dim, hidden_dim, num_classes)

# 5. Critère et optimiseur
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
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{epochs} – Loss: {total_loss/len(train_loader):.4f}")

# 7. Prédiction sur X_test
model.eval()
with torch.no_grad():
    logits = model(X_te)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    pred_labels = le.inverse_transform(preds)

# 8. Génération de la soumission
print("Saving submission2_2.csv...")
# Charger les Id du test
test_ids = pd.read_json('test.json', orient='records')['Id']
submission = pd.DataFrame({
    'Id': test_ids,
    'Category': pred_labels
})
submission.to_csv('submission2_2.csv', index=False)
print("✅ submission2_2.csv généré.")
