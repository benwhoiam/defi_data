import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

print("Début entraînement + prédiction avec PyTorch")

# -- 1. Charger données et encoders
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test  = np.load('X_test.npy')

vectorizer = joblib.load('vectorizer.joblib')
le         = joblib.load('label_encoder.joblib')

# -- 2. Convertir en tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# -- 3. Définir un modèle MLP (BOW) ou LSTM
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

input_dim   = X_train.shape[1]  # = MAX_LEN
hidden_dim  = 64
num_classes = y_train.shape[1]

model = SimpleMLP(input_dim, hidden_dim, num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -- 4. Entraînement
epochs = 15
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{epochs} – Loss: {total_loss/len(train_loader):.4f}")

# -- 5. Prédiction sur test
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs  = torch.sigmoid(logits).numpy()  # pour BCEWithLogitsLoss
    # argmax pour multi-classes
    preds = np.argmax(probs, axis=1)
    pred_labels = le.inverse_transform(preds)

# -- 6. Génération de la soumission
print("Saving submission file…")
test_ids = pd.read_json('test.json', orient='records')['Id']
submission = pd.DataFrame({
    'Id':       test_ids,
    'Category': pred_labels
})
submission.to_csv('submission2_2.csv', index=False)
print("✅ submission2_2.csv généré. ")
