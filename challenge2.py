import pandas as pd
import numpy as np
import json
import re
import spacy

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Masking, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Chargement et prétraitement des données

version_ = "V.2.0.0"
print("Version:", version_)

print("Loading training data...")
with open('train_mini.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data).fillna('')

print("Merging labels...")
labels_df = pd.read_csv('train_label.csv')
df = df.merge(labels_df, on='Id')

print("Cleaning and lemmatizing text data...")
nlp = spacy.load('en_core_web_sm')
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
    return ' '.join(tokens)

df['Clean'] = df['description'].apply(clean_text)

# 2. Encodage des labels
print("Encoding labels...")
le = LabelEncoder()
df['y'] = le.fit_transform(df['Category'])
num_classes = len(le.classes_)

# 3. Tokenisation & séquences
print("Tokenizing and creating sequences...")
MAX_VOCAB = 20000
MAX_LEN   = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Clean'])
sequences = tokenizer.texts_to_sequences(df['Clean'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = np.eye(num_classes)[df['y']]

# 4. Définition du modèle LSTM
print("Defining the LSTM model...")
EMBED_DIM = 128

model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Masking(mask_value=0.0),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Callbacks pour early stopping et sauvegarde du meilleur modèle
print("Setting up callbacks...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_lstm.h5', monitor='val_loss', save_best_only=True)
]

# 6. Entraînement
print("Training the model...")
history = model.fit(
    X, y,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)

# 7. Prédiction sur le test set
print("Loading test data...")
with open('test_mini.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data).fillna('')

print("Cleaning and lemmatizing test data...")
test_df['Clean'] = test_df['description'].apply(clean_text)

print("Tokenizing and padding test data...")
test_seq = tokenizer.texts_to_sequences(test_df['Clean'])
X_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print("Predicting categories for test data...")
pred_probs = model.predict(X_test)
pred_labels = pred_probs.argmax(axis=-1)
test_df['Predicted_Category'] = le.inverse_transform(pred_labels)

# 8. Préparation de la soumission
print("Preparing submission file...")
template = pd.read_csv('template_submissions.csv')
template['Category'] = template['Id'].map(test_df.set_index('Id')['Predicted_Category'])
template.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
print("Version:", version_)