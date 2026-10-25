import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))
from flask_app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Iris Flower Predictor</title>', response.data)

    def test_predict_valid_input(self):
        response = self.client.post('/predict', data={
            'sepal_length': 5.1, 'sepal_width': 3.5,
            'petal_length': 1.4, 'petal_width': 0.2})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Iris-setosa' in response.data
            or b'Iris-versicolor' in response.data
            or b'Iris-virginica' in response.data,
            'Response mein koi Iris species honi chahiye')

    def test_predict_invalid_input(self):
        response = self.client.post('/predict', data={
            'sepal_length': 'abc', 'sepal_width': 3.5,
            'petal_length': 1.4, 'petal_width': 0.2})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'Species:', response.data)


if __name__ == '__main__':
    unittest.main()