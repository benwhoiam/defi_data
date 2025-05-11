import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("Training LSTM with PyTorch")

# 1. Charger X, y et encoder
X = np.load('X.npy')
y = np.load('y.npy')
vectorizer = joblib.load('vectorizer.joblib')
le = joblib.load('label_encoder.joblib')

# 2. Convertir en PyTorch tensors
#    - Pour l’Embedding et LSTM, on a besoin d’indices entiers, 
#      mais ici X est un bag-of-words float32. Pour garder l’Embedding,
#      on peut directement traiter X comme floats via un simple MLP,
#      ou re-tokeniser. Ici on convertit en float tensor pour un MLP.
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Définir un modèle simple (MLP) ou adapter LSTM sur BOW
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

input_dim   = X.shape[1]      # = MAX_LEN
hidden_dim  = 64
num_classes = y.shape[1]

model = SimpleMLP(input_dim, hidden_dim, num_classes)
criterion = nn.BCEWithLogitsLoss()
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

# 5. Sauvegarde du modèle
torch.save(model.state_dict(), 'model_mlp.pth')
print("Model saved to model_mlp.pth")
