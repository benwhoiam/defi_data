import pandas as pd
import re
import spacy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
import json
from gensim.models import Word2Vec

version_ = "V.4.0.1"
print("Version:", version_)

# Load pre-trained Word2Vec model
print("Loading pre-trained Word2Vec model...")
pretrained_model_path = "models/pre_trained_model.pkl"
pretrained_model = Word2Vec.load(pretrained_model_path)

# Load training data
print("Loading training data...")
with open('train_mini.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data)

# Handle missing or invalid data in 'description'
print("Checking for problematic rows in 'description'...")
df['description'] = df['description'].fillna("")

# Load Spacy model
print("Loading Spacy model...")
nlp = spacy.load('en_core_web_sm')

# Define the clean_and_tokenize function
def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return "", []  # Return empty values for invalid or empty input
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    doc = nlp(text)  # Process text using SpaCy
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  # Lemmatize and remove stopwords
    return " ".join(tokens), tokens

# Clean and tokenize training data
print("Cleaning and tokenizing training data...")
results = df['description'].apply(clean_and_tokenize)
tuples = results.apply(lambda x: x if (isinstance(x, tuple) and len(x) == 2) else ("", []))
results_df = pd.DataFrame(tuples.tolist(), index=results.index, columns=['Clean', 'Tokens'])
df[['Clean', 'Tokens']] = results_df

# Load labels
print("Loading labels...")
labels_df = pd.read_csv('train_label.csv')
df = df.merge(labels_df, on='Id')

# Vectorize tokens using the pre-trained Word2Vec model
print("Vectorizing tokens using pre-trained Word2Vec model...")
def vectorize_tokens(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['W2V_Vector'] = df['Tokens'].apply(lambda tokens: vectorize_tokens(tokens, pretrained_model))

# Prepare training data
print("Preparing training data...")
X = np.vstack(df['W2V_Vector'].values)
y = df['Category'].values

# Train Neural Network model
print("Training Neural Network model...")
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
nn_model.fit(X, y)

# Load test data
print("Loading test data...")
with open('test_mini.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data)

# Clean and tokenize test data
print("Cleaning and tokenizing test data...")
test_df[['Clean', 'Tokens']] = test_df['description'].apply(lambda t: pd.Series(clean_and_tokenize(t)))

# Vectorize test data
print("Vectorizing test data using pre-trained Word2Vec model...")
test_df['W2V_Vector'] = test_df['Tokens'].apply(lambda tokens: vectorize_tokens(tokens, pretrained_model))
X_test_w2v = np.vstack(test_df['W2V_Vector'].values)

# Predict categories for test data
print("Predicting categories for test data...")
test_df['Predicted_Category_NN'] = nn_model.predict(X_test_w2v)

# Prepare submission file
print("Preparing submission file...")
template = pd.read_csv('template_submissions.csv')
template['Category'] = template['Id'].map(
    test_df.set_index('Id')['Predicted_Category_NN']
)
template.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'.")
print("Version:", version_)