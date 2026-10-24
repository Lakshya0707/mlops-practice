import os
import mlflow
from flask import Flask, render_template, request
from preprocessing_utility import preprocess, LABELS

mlflow.set_tracking_uri('https://dagshub.com/Lakshya0707/mlops-practice.mlflow')
MODEL_URI = 'models:/iris_model@staging'
model = mlflow.pyfunc.load_model(MODEL_URI)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        vals = [float(request.form[k]) for k in
                ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        features = preprocess(*vals)
        pred = int(model.predict(features)[0])
        return render_template('index.html', prediction=LABELS[pred])
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)