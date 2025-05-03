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
df = pd.DataFrame(train_data)
print("Checking for problematic rows in 'description'...")
print(df['description'].isnull().sum(), "rows are null in 'description'")
print(df['description'].head())

df['description'] = df['description'].fillna("")

print("Loading Spacy model...")
nlp = spacy.load('en_core_web_sm')

print("Cleaning and tokenizing training data...")

# Handle missing or invalid data in 'description'
df['description'] = df['description'].fillna("")

# Define the clean_and_tokenize function with error handling
def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return "", []  # Return empty values for invalid or empty input
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    doc = nlp(text)  # Process text using SpaCy
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  # Lemmatize and remove stopwords
    return " ".join(tokens), tokens

# Apply the function to the 'description' column
print("Cleaning and tokenizing training data...")
results = df['description'].apply(clean_and_tokenize)
# 1. On transforme chaque tuple en une liste de taille 2, en remplaçant les invalides par ("",[])
tuples = results.apply(lambda x: x if (isinstance(x, tuple) and len(x)==2) else ("","[]"))

# 2. On crée un DataFrame à partir de ces tuples, en préservant l'index d'origine
results_df = pd.DataFrame(tuples.tolist(), 
                          index=results.index, 
                          columns=['Clean', 'Tokens'])

# 3. On l'ajoute directement à ton DataFrame principal
df[['Clean', 'Tokens']] = results_df

print(df.columns)
print(df.head())



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

print("Training Logistic Regression model...")
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X, y)


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
template = pd.read_csv('template_submissions.csv')
template['Category'] = template['Id'].map(
    test_df.set_index('Id')['Predicted_Category_W2V']
)
template.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'.")
