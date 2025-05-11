import pandas as pd
import numpy as np
import json
import re
import spacy
import joblib
from sklearn.preprocessing    import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
print("V2_1.1.0 – Prétraitement train + test et sauvegarde")

# -- 1. Chargement train
print("Loading training data...")
with open('train_mini.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data).fillna('')

print("Merging train labels...")
labels_df = pd.read_csv('train_label.csv')
df = df.merge(labels_df, on='Id')

# -- 2. Chargement test
print("Loading test data...")
with open('test_mini.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data).fillna('')

# -- 3. Nettoyage & lemmatisation
print("Cleaning and lemmatizing text data...")
nlp = spacy.load('en_core_web_sm')
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    return ' '.join(tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop)

df['Clean']      = df['description'].apply(clean_text)
test_df['Clean'] = test_df['description'].apply(clean_text)

# -- 4. Encodage des labels train
print("Encoding train labels...")
le = LabelEncoder()
df['y'] = le.fit_transform(df['Category'])
num_classes = len(le.classes_)

# -- 5. Vectorisation train + test
print("Vectorizing train + test (CountVectorizer)...")
MAX_VOCAB = 20000
MAX_LEN   = 100

vectorizer = CountVectorizer(max_features=MAX_VOCAB, stop_words='english')
X_train = vectorizer.fit_transform(df['Clean']).toarray()
X_test  = vectorizer.transform(test_df['Clean']).toarray()

# -- 6. Truncate/pad train + test
def pad_trunc(mat, max_len):
    if mat.shape[1] > max_len:
        return mat[:, :max_len]
    elif mat.shape[1] < max_len:
        return np.pad(mat, ((0,0),(0, max_len - mat.shape[1])), mode='constant')
    else:
        return mat

X_train = pad_trunc(X_train, MAX_LEN)
X_test  = pad_trunc(X_test,  MAX_LEN)

# -- 7. One-hot des labels train
y_train = np.eye(num_classes)[df['y']]

# -- 8. Sauvegarde de tous les artefacts
print("Saving vectorizer, label encoder, X_train.npy, y_train.npy, X_test.npy …")
joblib.dump(vectorizer,    'vectorizer.joblib')
joblib.dump(le,            'label_encoder.joblib')
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy',  X_test)

print("✅ Prétraitement terminé et fichiers sauvegardés.")
