import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the dataset

version_ = "V.3.0.0"
print("Version:", version_)

print("Loading dataset...")
data = pd.read_json('train_mini.json').set_index('Id')
print("Dataset loaded successfully!")

# Preprocessing
print("Preprocessing data...")
X = data['description']  # Text descriptions
y = data['gender']       # Labels (replace with job categories if available)
print(f"Number of samples: {len(X)}")

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
print("Labels encoded successfully!")

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
X_test = test_data['description']  # Assuming the column name is 'description'
X_test_tokenized = tokenizer.texts_to_sequences(X_test)  # Tokenize the text
X_test_padded = pad_sequences(X_test_tokenized, maxlen=100)  # Pad the sequences
print("Test data preprocessed successfully!")

# Predict the categories for test data
print("Generating predictions for test data...")
predictions = model.predict(X_test_padded)
predicted_classes = predictions.argmax(axis=1)  # Get the class with the highest probability
print("Predictions generated successfully!")

# Map predicted class indices back to their original labels
print("Mapping predictions to labels...")
predicted_labels = label_encoder.inverse_transform(predicted_classes)
print("Predictions mapped to labels successfully!")

# Save the predictions to submissions3.csv
print("Saving predictions to submissions3.csv...")
submissions3 = pd.DataFrame({
    'Id': test_data.index,
    'Category': predicted_labels
})
submissions3.to_csv('submissions3.csv', index=False)
print("Predictions saved to submissions3.csv successfully!")