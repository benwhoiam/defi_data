import pandas as pd
import json
import matplotlib.pyplot as plt

# Load JSON data
with open('d:/enac/defi_data/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert JSON to DataFrame
df_json = pd.DataFrame({
    'Id': [int(i) for i in data['Id'].values()],
    'gender': [data['gender'][k] for k in data['gender']],
})

# Load CSV data
df_csv = pd.read_csv('d:/enac/defi_data/train_label.csv', usecols=['Id', 'Category'])

# Merge on Id
df = pd.merge(df_json, df_csv, on='Id')

# Calculate gender percentage per category
gender_counts = df.groupby(['Category', 'gender']).size().unstack(fill_value=0)
gender_percent = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100

# Plot
gender_percent.plot(kind='bar', stacked=True)
plt.ylabel('Percentage')
plt.title('Percentage of Men and Women per Category')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.tight_layout()
plt.show()