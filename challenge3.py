import pandas as pd
import re
import spacy
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load SpaCy model for text preprocessing
print("Loading SpaCy model...")
nlp = spacy.load('en_core_web_sm')

# Function to clean and preprocess text
def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return ""  # Return empty string for invalid or empty input
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    doc = nlp(text)  # Process text using SpaCy
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  # Lemmatize and remove stopwords
    return " ".join(tokens)

# Load the dataset
version_ = "V.3.0.6"
print("Version:", version_)
print("Loading dataset...")
data = pd.read_json('train_mini.json').set_index('Id')
print("Dataset loaded successfully!")

# Preprocessing
print("Preprocessing data...")
data['Cleaned_Description'] = data['description'].apply(clean_and_tokenize)  # Clean and tokenize descriptions
X = data['Cleaned_Description']  # Use cleaned descriptions for training
print(f"Number of samples: {len(X)}")

# Load category label indices for each training sample
print("Loading training labels...")
labels = pd.read_csv('train_label_mini.csv').set_index('Id')

# Merge labels with training data
data = data.merge(labels, left_index=True, right_index=True)
print("Training labels merged successfully!")

# Load category index-to-name mapping (optional, for future use)
category_name_map = pd.read_csv('categories_string.csv', header=None, index_col=1)[0].to_dict()

# For training, use label index directly
y_encoded = data['Category']  # already numeric label
y_categorical = to_categorical(y_encoded)
# Debug: Print unmapped values
unmapped = data[data['Category'].isnull()]
if not unmapped.empty:
    print("Warning: Some descriptions could not be mapped. Unmapped values:")
    print(unmapped[['description', 'Cleaned_Description']])

# Drop rows with missing Category values
data = data.dropna(subset=['Category'])
if data.empty:
    raise ValueError("No valid data left after mapping categories. Check your input files.")
print("Categories mapped successfully!")

# Encode job categories
print("Encoding job categories...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Category'])  # Encode the mapped categories
y_categorical = to_categorical(y_encoded)
print("Job categories encoded successfully!")

# Tokenize and pad text data
print("Tokenizing and padding text data...")
tokenizer = Tokenizer(num_words=5000)  # Limit vocabulary size
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_tokenized, maxlen=100)  # Limit sequence length
print("Text data tokenized and padded successfully!")

# Build the neural network model
print("Building the neural network model...")
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),  # Embedding layer
    GlobalAveragePooling1D(),  # Reduces dimensionality
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer
])
print("Model built successfully!")

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully!")

# Train the model on all available data
print("Training the model on all available data...")
model.fit(X_padded, y_categorical, epochs=10, batch_size=32, verbose=1)
print("Model training completed!")

# Load the test dataset
print("Loading test dataset...")
test_data = pd.read_json('test_mini.json').set_index('Id')  # Replace with the actual file path if needed
print(f"Number of test samples: {len(test_data)}")

# Preprocess the text data in test_mini.json
print("Preprocessing test data...")
test_data['Cleaned_Description'] = test_data['description'].apply(clean_and_tokenize)  # Clean and tokenize descriptions
X_test = test_data['Cleaned_Description']
X_test_tokenized = tokenizer.texts_to_sequences(X_test)  # Tokenize the text
X_test_padded = pad_sequences(X_test_tokenized, maxlen=100)  # Pad the sequences
print("Test data preprocessed successfully!")

# Predict the categories for test data
print("Generating predictions for test data...")
predictions = model.predict(X_test_padded)
predicted_classes = predictions.argmax(axis=1)  # Get the class with the highest probability
print("Predictions generated successfully!")

# Map predicted class indices back to their original job categories
print("Mapping predictions to job categories...")
predicted_categories = label_encoder.inverse_transform(predicted_classes)
print("Predictions mapped to job categories successfully!")

# Save the predictions to submissions3.csv
print("Saving predictions to submissions3.csv...")
submissions3 = pd.DataFrame({
    'Id': test_data.index,
    'Category': predicted_categories
})
submissions3.to_csv('submissions3.csv', index=False)
print("Predictions saved to submissions3.csv successfully!")