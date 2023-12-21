import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


app = Flask(__name__)
CORS(app)

# Charger le modèle
model = load_model('models/text_generation_model1.h5')


def init():
    file = 'dataset/dataset.csv'
    df = pd.read_csv(file)
    df['class'] = df['class'].replace({0: 'Human', 1: 'AI'})

    X = df['article'].values
    y = df['class'].values
    vocab_size = 1000  # Taille du vocabulaire, à ajuster

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tokenization
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)
    text = data['text']

    tokenizer = init()
    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(sequence, maxlen=1500, padding='post', truncating='post')

    # Faire la prédiction
    result = model.predict(np.array(padded_sequence))
    prediction = 'AI' if result[0][0] < 0.5 else 'Human'

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(port=8080, debug=False)
