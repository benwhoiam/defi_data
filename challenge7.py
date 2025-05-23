import json
import pandas as pd
import re
import spacy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, make_scorer, f1_score

# 1) Chargement et préparation
with open('train.json','r',encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data).fillna({'description':""})

labels = pd.read_csv('train_label.csv')
df = df.merge(labels, on='Id')

# 2) Tokenisation / nettoyage
nlp = spacy.load('en_core_web_sm')
def clean_and_tokenize(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text).lower().strip()
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop)

df['Clean'] = df['description'].apply(clean_and_tokenize)

# 3) Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    df['Clean'], df['Category'],
    test_size=0.2, stratify=df['Category'], random_state=42
)

# 4) Pipeline + GridSearchCV
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC(dual=False, max_iter=5000))
])

param_grid = {
    # TF-IDF params
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [3, 5, 10],
    "tfidf__max_df": [0.8, 0.9, 1.0],
    # SVM params
    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "clf__class_weight": [None, "balanced"]
}

# Utiliser macro-F1 pour équilibrer toutes les classes
scorer = make_scorer(f1_score, average="macro")

grid = GridSearchCV(
    pipeline, param_grid,
    cv=5, n_jobs=-1, scoring=scorer, verbose=1
)
grid.fit(X_train, y_train)



# 5) Évaluation sur la validation
y_val_pred = grid.predict(X_val)
print("=== Rapport sur le set de validation ===")
print(classification_report(y_val, y_val_pred))

# 6) Appliquer au jeu de test
with open('test.json','r',encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data).fillna({'description':""})
test_df['Clean'] = test_df['description'].apply(clean_and_tokenize)

preds = grid.predict(test_df['Clean'])

# 7) Générer la soumission
template = pd.read_csv('template_submissions.csv')
mapping = dict(zip(test_df['Id'], preds))
template['Category'] = template['Id'].map(mapping)
template.to_csv('submission_svm_tuned.csv', index=False)
print(" Soumission enregistrée : submission_svm_tuned.csv")

