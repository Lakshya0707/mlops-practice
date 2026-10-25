import os
import sys
import unittest

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))
from preprocessing_utility import preprocess, LABELS


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.getenv('MLFLOW_TRACKING_USERNAME') or not os.getenv('MLFLOW_TRACKING_PASSWORD'):
            raise EnvironmentError('MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD set nahi hai')

        mlflow.set_tracking_uri('https://dagshub.com/Lakshya0707/mlops-practice.mlflow')

        # Registry se staging alias wala model
        cls.model_uri = 'models:/iris_model@staging'
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Holdout test data (feature engineering ke baad wala)
        cls.holdout = pd.read_csv('data/interim/test_feature_engineering.csv')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        features = preprocess(5.1, 3.5, 1.4, 0.2)

        # Input columns training data jaise hi hone chahiye (naam aur order)
        expected_cols = list(self.holdout.drop(columns=['Species']).columns)
        self.assertEqual(list(features.columns), expected_cols)

        prediction = self.model.predict(features)
        self.assertEqual(len(prediction), features.shape[0])
        self.assertEqual(len(prediction.shape), 1)
        self.assertIn(int(prediction[0]), LABELS)

    def test_model_performance(self):
        X = self.holdout.drop(columns=['Species'])
        y = self.holdout['Species']
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

        expected = 0.85
        self.assertGreaterEqual(accuracy, expected, f'Accuracy kam hai: {accuracy:.2f}')
        self.assertGreaterEqual(precision, expected, f'Precision kam hai: {precision:.2f}')
        self.assertGreaterEqual(recall, expected, f'Recall kam hai: {recall:.2f}')
        self.assertGreaterEqual(f1, expected, f'F1 kam hai: {f1:.2f}')


if __name__ == '__main__':
    unittest.main()