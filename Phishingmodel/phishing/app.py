from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('rf_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        url_vectorized = vectorizer.transform([url])
        prediction = model.predict(url_vectorized)
        prediction_label = 'Angler Phishing' if prediction[0] == 1 else 'Safe'
        return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
