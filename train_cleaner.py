import pandas as pd
import json
import spacy
import re

nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Nettoyage train
with open('train.json','r',encoding='utf-8') as f:
    train = pd.DataFrame(json.load(f))
train['description']=train['description'].fillna("")
train['Clean']=train['description'].apply(clean_text)
train.to_csv('train_cleaned.csv',index=False)

# Nettoyage test
with open('test.json','r',encoding='utf-8') as f:
    test = pd.DataFrame(json.load(f))
test['description']=test['description'].fillna("")
test['Clean']=test['description'].apply(clean_text)
test.to_csv('test_cleaned.csv',index=False)

print("Cleaned CSV générés.")
