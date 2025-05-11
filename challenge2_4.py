import json
import re
import numpy as np
import pandas as pd
import spacy
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
print("V2_1.2.1 – Prétraitement séquences pour LSTM (sans collections)")

# Paramètres
MAX_VOCAB = 20000
MAX_LEN   = 100

# 1. Charger train + test
print("Loading data...")
with open('train_mini.json','r',encoding='utf-8') as f:
    df_train = pd.DataFrame(json.load(f)).fillna('')
with open('test_mini.json','r',encoding='utf-8') as f:
    df_test = pd.DataFrame(json.load(f)).fillna('')

labels_df = pd.read_csv('train_label.csv')
df_train = df_train.merge(labels_df, on='Id')

# 2. Nettoyage & tokenisation (lemmatisation)
print("Cleaning & tokenizing...")
nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
def clean_tokenize(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    return [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]

df_train['Tokens'] = df_train['description'].apply(clean_tokenize)
df_test ['Tokens'] = df_test ['description'].apply(clean_tokenize)

# 3. Construire le vocabulaire (stoi) manuellement
print("Building vocabulary (manual count)...")
freq = {}
for toks in df_train['Tokens']:
    for w in toks:
        freq[w] = freq.get(w, 0) + 1

# Trier par fréquence décroissante et garder top MAX_VOCAB-2
most_common = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:MAX_VOCAB-2]
itos = ['<PAD>','<UNK>'] + [w for w,_ in most_common]
stoi = {w:i for i,w in enumerate(itos)}

# 4. Convertir tokens → indices + pad/truncate
print("Converting to sequences...")
def tokens_to_seq(tokens):
    seq = [stoi.get(w, stoi['<UNK>']) for w in tokens]
    if len(seq) < MAX_LEN:
        seq += [stoi['<PAD>']] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq

X_train_seq = np.array([tokens_to_seq(t) for t in df_train['Tokens']], dtype=np.int64)
X_test_seq  = np.array([tokens_to_seq(t) for t in df_test ['Tokens']], dtype=np.int64)

# 5. Encoder les labels (entiers)
print("Encoding labels...")
le = LabelEncoder()
y_train = le.fit_transform(df_train['Category'].values)

# 6. Sauvegarder artefacts
print("Saving artifacts...")
joblib.dump(stoi,              'vocab_stoi.pkl')
joblib.dump(le,                'label_encoder.pkl')
np.save('X_train_seq.npy',      X_train_seq)
np.save('y_train.npy',          y_train)
np.save('X_test_seq.npy',       X_test_seq)

print("✅ Prétraitement terminé. Fichiers générés :")
print("   vocab_stoi.pkl, label_encoder.pkl,")
print("   X_train_seq.npy, y_train.npy, X_test_seq.npy")
