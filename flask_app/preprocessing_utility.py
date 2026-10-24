import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE, '..', 'artifacts', 'scaler.pkl')

FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
LABELS = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

scaler = joblib.load(SCALER_PATH)

def preprocess(sepal_length, sepal_width, petal_length, petal_width) -> pd.DataFrame:
    df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=FEATURES)
    scaled = pd.DataFrame(scaler.transform(df), columns=FEATURES)
    scaled['PetalInteraction'] = scaled['PetalLengthCm'] * scaled['PetalWidthCm']
    scaled['SepalInteraction'] = scaled['SepalLengthCm'] * scaled['SepalWidthCm']
    return scaled