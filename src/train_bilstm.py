import os
import yaml
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# =============================
# Load configuration
# =============================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

bilstm_config = config["bilstm"]

# =============================
# Load Data
# =============================
train_df = pd.read_csv("Data/processed/train.csv")
val_df = pd.read_csv("Data/processed/val.csv")
test_df = pd.read_csv("Data/processed/test.csv")

# Using 'review' and 'sentiment' instead of 'text' and 'label'
X_train = train_df["review"].astype(str).tolist()
y_train = train_df["sentiment"].values

X_val = val_df["review"].astype(str).tolist()
y_val = val_df["sentiment"].values

X_test = test_df["review"].astype(str).tolist()
y_test = test_df["sentiment"].values

# =============================
# Tokenizer
# =============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# =============================
# Train Word2Vec embeddings
# =============================
sentences = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=bilstm_config["vector_size"], window=5, min_count=1, workers=4)
embedding_matrix = np.zeros((vocab_size, bilstm_config["vector_size"]))

for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# =============================
# Data Generator
# =============================
class DataGenerator(Sequence):
    def __init__(self, texts, labels, tokenizer, batch_size, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.texts[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        sequences = self.tokenizer.texts_to_sequences(batch_x)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return np.array(padded), np.array(batch_y)

train_generator = DataGenerator(X_train, y_train, tokenizer, bilstm_config["batch_size"], bilstm_config["max_len"])
val_generator = DataGenerator(X_val, y_val, tokenizer, bilstm_config["batch_size"], bilstm_config["max_len"])
test_generator = DataGenerator(X_test, y_test, tokenizer, bilstm_config["batch_size"], bilstm_config["max_len"])

# =============================
# Build BiLSTM Model
# =============================
input_layer = Input(shape=(bilstm_config["max_len"],))
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=bilstm_config["vector_size"],
                            weights=[embedding_matrix],
                            input_length=bilstm_config["max_len"],
                            trainable=False)(input_layer)

x = Bidirectional(LSTM(bilstm_config["lstm_units_1"], return_sequences=True,
                       kernel_regularizer=l2(bilstm_config["l2_reg"])))(embedding_layer)
x = Dropout(bilstm_config["dropout"])(x)
x = Bidirectional(LSTM(bilstm_config["lstm_units_2"],
                       kernel_regularizer=l2(bilstm_config["l2_reg"])))(x)
x = Dropout(bilstm_config["dropout"])(x)

x = Dense(bilstm_config["dense_units_1"], activation="relu",
          kernel_regularizer=l2(bilstm_config["l2_reg"]))(x)
x = Dropout(bilstm_config["dropout"])(x)
x = Dense(bilstm_config["dense_units_2"], activation="relu",
          kernel_regularizer=l2(bilstm_config["l2_reg"]))(x)
output_layer = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=bilstm_config["learning_rate"])
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# =============================
# Train Model
# =============================
model.fit(train_generator,
          validation_data=val_generator,
          epochs=bilstm_config["epochs"])

# =============================
# Save Model and Tokenizer
# =============================
os.makedirs("models", exist_ok=True)
model.save("models/bilstm_model.h5")

with open("models/tokenizer_bilstm.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

w2v_model.save("models/word2vec.model")

# =============================
# Save Predictions
# =============================
pred_test = model.predict(test_generator)

os.makedirs("Data/predictions", exist_ok=True)
pred_df = pd.DataFrame(pred_test, columns=["prediction"])
pred_df.to_csv("Data/predictions/bilstm_preds.csv", index=False)

print("BiLSTM model, tokenizer, Word2Vec, and predictions saved successfully!")

