function predict() {
    // Récupérer le texte saisi par l'utilisateur
    const userInput = document.getElementById('textInput').value;

    // Envoyer une requête POST à l'API Flask
    fetch('http://127.0.0.1:8080/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: userInput,
        }),
    })
        .then(response => response.json())
        .then(data => {
            // Mettre à jour l'interface avec le résultat de la prédiction
            document.getElementById('predictionResult').innerText = `Prediction: ${data.prediction}`;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
