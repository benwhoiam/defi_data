import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import json
import numpy as np

print("Loading training data...")
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame.from_dict(train_data, orient='index', columns=['description', 'Gender']).reset_index().rename(columns={'index': 'Id'})

print("Loading Spacy model...")
nlp = spacy.load('en_core_web_sm')

def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return "", []  # Return empty values for invalid or empty input
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    doc = nlp(text)  # Process text using SpaCy
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  # Lemmatize and remove stopwords
    return " ".join(tokens), tokens

print("Cleaning and tokenizing training data...")
df[['Clean', 'Tokens']] = df['description'].apply(lambda t: pd.Series(clean_and_tokenize(t)))

print("Training Word2Vec model...")
model = Word2Vec(
    sentences=df['Tokens'],
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)
model.build_vocab(df['Tokens'])
model.train(df['Tokens'], total_examples=model.corpus_count, epochs=10)

print("Loading labels...")
labels_df = pd.read_csv('train_label.csv')
df = df.merge(labels_df, on='Id')

print("Vectorizing tokens using Word2Vec...")
def vectorize_tokens(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['W2V_Vector'] = df['Tokens'].apply(lambda tokens: vectorize_tokens(tokens, model))

print("Preparing training and testing data...")
X = np.vstack(df['W2V_Vector'].values)
y = df['Category'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression model...")
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train, y_train)

print("Evaluating model...")
y_pred = clf_w2v.predict(X_test)
print(classification_report(y_test, y_pred))

print("Predicting categories for training data...")
df['Predicted_Category'] = clf_w2v.predict(X)

print("Loading test data...")
with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data)

print("Cleaning and tokenizing test data...")
test_df[['Clean', 'Tokens']] = test_df['description'].apply(lambda t: pd.Series(clean_and_tokenize(t)))

print("Vectorizing test data using Word2Vec...")
test_df['W2V_Vector'] = test_df['Tokens'].apply(lambda tokens: vectorize_tokens(tokens, model))
X_test_w2v = np.vstack(test_df['W2V_Vector'].values)

print("Predicting categories for test data...")
test_df['Predicted_Category_W2V'] = clf_w2v.predict(X_test_w2v)

print("Preparing submission file...")
template = pd.read_csv('template_submission.csv')
template['Category'] = template['Id'].map(
    test_df.set_index('Id')['Predicted_Category_W2V']
)
template.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'.")
