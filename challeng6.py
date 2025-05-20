import json
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Chargement des données
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data)
df['description'] = df['description'].fillna("")

# SpaCy 2.x
nlp = spacy.load('en_core_web_sm')

def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)

df['Clean'] = df['description'].apply(clean_and_tokenize)

# Chargement des labels
labels_df = pd.read_csv('train_label.csv')
df = df.merge(labels_df, on='Id')

# Vectorisation TF-IDF
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=5, norm='l2')
X_tfidf = tfidf.fit_transform(df['Clean'])
y = df['Category']

X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, y, test_size=0.2, stratify=y, random_state=42
)

# Modèle SVM linéaire
svm_clf = LinearSVC(C=1.0, max_iter=5000)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_val)

print(classification_report(y_val, y_pred))

# Prédiction test
with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data)
test_df['Clean'] = test_df['description'].fillna("").apply(clean_and_tokenize)
X_test = tfidf.transform(test_df['Clean'])
preds = svm_clf.predict(X_test)

template = pd.read_csv('template_submissions.csv')
template['Category'] = template['Id'].map(dict(zip(test_df['Id'], preds)))
template.to_csv('submission_svm.csv', index=False)
print("Fichier de soumission enregistré : 'submission_svm.csv'")
