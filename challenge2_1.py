import pandas as pd
import numpy as np
import json
import re
import spacy
import joblib  # Pour la sauvegarde
print( "V.2_1.0.0")

# Workaround for NumPy versions where np.object is removed
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Clean'])
sequences = tokenizer.texts_to_sequences(df['Clean'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = np.eye(num_classes)[df['y']]

# Sauvegarde des objets
print("Saving tokenizer, label encoder, X, and y...")

# Sauvegarder le tokenizer et le label encoder
joblib.dump(tokenizer, 'tokenizer.joblib')
joblib.dump(le, 'label_encoder.joblib')

# Sauvegarder X et y sous forme de fichiers .npy
np.save('X.npy', X)
np.save('y.npy', y)

print("Objects saved successfully!")
