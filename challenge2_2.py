import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Chargement des données
X_tensor = torch.load('X_tensor.pth')
y_tensor = torch.load('y_tensor.pth')

# Définition du modèle LSTM avec PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = self.dropout(hn[-1])
        output = self.fc(lstm_out)
        return output

# Paramètres du modèle
MAX_VOCAB = 20000
EMBED_DIM = 128
HIDDEN_DIM = 64
num_classes = y_tensor.shape[1]

# Initialisation du modèle
model = LSTMModel(MAX_VOCAB, EMBED_DIM, HIDDEN_DIM, num_classes)

# Fonction de perte et optimiseur
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Sauvegarde du modèle entraîné
torch.save(model.state_dict(), 'lstm_model.pth')

print("Model saved to lstm_model.pth")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Chargement des données
X_tensor = torch.load('X_tensor.pth')
y_tensor = torch.load('y_tensor.pth')

# Définition du modèle LSTM avec PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = self.dropout(hn[-1])
        output = self.fc(lstm_out)
        return output

# Paramètres du modèle
MAX_VOCAB = 20000
EMBED_DIM = 128
HIDDEN_DIM = 64
num_classes = y_tensor.shape[1]

# Initialisation du modèle
model = LSTMModel(MAX_VOCAB, EMBED_DIM, HIDDEN_DIM, num_classes)

# Fonction de perte et optimiseur
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Sauvegarde du modèle entraîné
torch.save(model.state_dict(), 'lstm_model.pth')

print("Model saved to lstm_model.pth")
