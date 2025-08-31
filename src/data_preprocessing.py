import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# 1. Load raw data
path = 'Data/raw/IMDB Dataset.csv'
df = pd.read_csv(path)

# 2. Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# 3. Save full cleaned data
df[['review', 'clean_review', 'sentiment']].to_csv("Data/processed/cleaned_imdb.csv", index=False)
print("✅ Full cleaned dataset saved to Data/processed/cleaned_imdb.csv")

# 4. Split into Train/Val/Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['sentiment'])

# 5. Save splits
train_df.to_csv("Data/processed/train.csv", index=False)
val_df.to_csv("Data/processed/val.csv", index=False)
test_df.to_csv("Data/processed/test.csv", index=False)

print(f"✅ Train size: {len(train_df)}")
print(f"✅ Validation size: {len(val_df)}")
print(f"✅ Test size: {len(test_df)}")
print("✅ Splits saved to Data/processed/")
