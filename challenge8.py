import json
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Chargement et préparation des données d'entraînement
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df_train = pd.DataFrame(train_data)
df_train['description'] = df_train['description'].fillna("")

# 2. Chargement des labels
labels_df = pd.read_csv('train_label.csv')  # colonnes : Id, Category
df_train = df_train.merge(labels_df, on='Id')

# 3. Initialisation de SpaCy et fonction de nettoyage + lemmatisation
nlp = spacy.load('en_core_web_lg')

def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)

df_train['Clean'] = df_train['description'].apply(clean_and_tokenize)

# 4. Vectorisation TF-IDF sur l'entraînement
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=5,
    norm='l2'
)
X_tfidf = tfidf.fit_transform(df_train['Clean'])
y = df_train['Category']

# 5. Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 6. Entraînement du modèle RandomForest
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# 7. Évaluation sur la validation
y_pred = rf_clf.predict(X_val)
print("=== Rapport de classification sur la validation ===")
print(classification_report(y_val, y_pred))


# ────────────────────────────
# 8. Chargement et préparation des données de test
# ────────────────────────────
with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
df_test = pd.DataFrame(test_data)
df_test['description'] = df_test['description'].fillna("")
df_test['Clean'] = df_test['description'].apply(clean_and_tokenize)

# 9. Vectorisation TF-IDF sur le test
X_test = tfidf.transform(df_test['Clean'])

# 10. Prédiction
preds = rf_clf.predict(X_test)

# 11. Préparation du fichier de soumission
template = pd.read_csv('template_submissions.csv')  # colonnes : Id, Category
# On remplace la colonne Category par nos prédictions, en s'assurant de l'alignement sur l'Id
mapping = dict(zip(df_test['Id'], preds))
template['Category'] = template['Id'].map(mapping)

# 12. Sauvegarde
output_path = 'submission_rf.csv'
template.to_csv(output_path, index=False)
print(f"Fichier de soumission enregistré : '{output_path}'")

