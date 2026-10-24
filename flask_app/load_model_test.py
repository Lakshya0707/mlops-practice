import sys, os
import unittest
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))
from preprocessing_utility import preprocess, LABELS

mlflow.set_tracking_uri('https://dagshub.com/Lakshya0707/mlops-practice.mlflow')

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mlflow.pyfunc.load_model('models:/iris_model@staging')

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_prediction_valid(self):
        pred = self.model.predict(preprocess(5.1, 3.5, 1.4, 0.2))
        self.assertEqual(len(pred), 1)
        self.assertIn(int(pred[0]), LABELS)

    def test_setosa_sample(self):
        pred = self.model.predict(preprocess(5.1, 3.5, 1.4, 0.2))
        self.assertEqual(int(pred[0]), 0)

if __name__ == '__main__':
    unittest.main()