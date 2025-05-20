import json
import re
import pandas as pd
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Chargement et nettoyage des données
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data)
df['description'] = df['description'].fillna("")

# 2. Chargement du modèle lg
# Assure-toi d'avoir préalablement : python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')

# 3. Fonction de nettoyage + récupération du vecteur SpaCy
def clean_and_vectorize(text):
    # Nettoyage léger (on conserve juste pour illustrer)
    txt = re.sub(r'<[^>]+>', ' ', text)
    txt = re.sub(r'http\S+', ' ', txt)
    txt = txt.lower().strip()
    # Passage dans spaCy
    doc = nlp(txt)
    # doc.vector est la moyenne des vecteurs de tokens (300 dims)
    return doc.vector

# 4. Création de la matrice de features
vectors = np.vstack(df['description'].apply(clean_and_vectorize).values)
y = pd.read_csv('train_label.csv').set_index('Id').loc[df['Id'], 'Category'].values

# 5. Train/Test split
X_train, X_val, y_train, y_val = train_test_split(
    vectors, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Entraînement du modèle
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train, y_train)

# 7. Évaluation
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

# 8. Prédiction sur le test set
with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data)
test_vectors = np.vstack(test_df['description'].apply(clean_and_vectorize).values)

preds = clf.predict(test_vectors)

# 9. Génération de la soumission
template = pd.read_csv('template_submissions.csv')
template['Category'] = template['Id'].map(dict(zip(test_df['Id'], preds)))
template.to_csv('submission_spacy_lg.csv', index=False)
print("Saved submission as 'submission_spacy_lg.csv'")
