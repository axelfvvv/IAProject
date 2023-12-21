import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


file = 'dataset.csv'
df = pd.read_csv(file)
df['class'] = df['class'].replace({0: 'Human', 1: 'AI'})
df.to_csv('dataset_modified.csv', index=False)

X = df['article'].values
y = df['class'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


vocab_size = 1000  # Taille du vocabulaire, ajustez selon votre dataset
embedding_dim = 50  # Dimension de l'espace d'embedding, ajustez selon votre choix
max_sequence_length = 1500  # Longueur maximale d'une séquence, ajustez selon votre choix

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Padding pour assurer que toutes les séquences ont la même longueur
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Encoder les labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Définition du modèle
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
epochs = 10  # Ajustez le nombre d'époques en fonction de votre dataset
batch_size = 32  # Ajustez la taille du lot en fonction de votre choix
history = model.fit(X_train_padded, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(X_val_padded, y_val_encoded))

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print(f'\nAccuracy on test set: {accuracy}')


# Sauvegarder le modèle
model.save('models/text_generation_model.h5')


