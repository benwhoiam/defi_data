import pandas as pd
import numpy as np
import json
import re
import spacy
import joblib  # Pour la sauvegarde
print("V2_1.0.0")
# 1. Chargement et prétraitement des données
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
    text = re.sub(r'<[^>]+>', ' ', text)  # Supprimer les balises HTML
    text = re.sub(r'http\S+', ' ', text)  # Supprimer les URLs
    text = text.lower().strip()  # Mettre en minuscule et enlever les espaces
    doc = nlp(text)  # Appliquer spaCy pour la lemmatisation
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]  # Lemmatisation et suppression des mots vides
    return ' '.join(tokens)

df['Clean'] = df['description'].apply(clean_text)

# 2. Encodage des labels
print("Encoding labels...")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['y'] = le.fit_transform(df['Category'])
num_classes = len(le.classes_)

# 3. Tokenisation & séquences
print("Tokenizing and creating sequences...")
from sklearn.feature_extraction.text import CountVectorizer

MAX_VOCAB = 20000
MAX_LEN = 100

# Use CountVectorizer to tokenize and create a representation of words
vectorizer = CountVectorizer(max_features=MAX_VOCAB, stop_words='english')
X = vectorizer.fit_transform(df['Clean']).toarray()

# Padding or truncating the matrix X to have a uniform length of MAX_LEN
if X.shape[1] > MAX_LEN:
    # Truncate sequences to MAX_LEN
    X = X[:, :MAX_LEN]
else:
    # Pad sequences to MAX_LEN
    X = np.pad(X, ((0, 0), (0, MAX_LEN - X.shape[1])), 'constant', constant_values=0)

# Create labels y as one-hot encoding
y = np.eye(num_classes)[df['y']]

# Save objects
print("Saving tokenizer, label encoder, X, and y...")

# Save the tokenizer and label encoder
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')

# Save X and y as .npy files
np.save('X.npy', X)
np.save('y.npy', y)

print("Objects saved successfully!")

