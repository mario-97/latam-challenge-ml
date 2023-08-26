import unittest

from fastapi.testclient import TestClient
from challenge import app

from unittest.mock import patch
import numpy as np

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        """ response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]}) """

        with patch("xgboost.XGBClassifier") as mock_model:
            mock_model.return_value.predict.return_value = np.array([0])

            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        with patch("xgboost.XGBClassifier") as mock_model:
            mock_model.return_value.predict.return_value = np.array([0])

            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)  # Cambia el código de respuesta esperado
        self.assertEqual(response.json(), {"detail": "Unknown column found"})  # Cambia el resultado esperado

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        with patch("xgboost.XGBClassifier") as mock_model:
            mock_model.return_value.predict.return_value = np.array([0])

            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)  # Cambia el código de respuesta esperado
        self.assertEqual(response.json(), {"detail": "Unknown column found"})  # Cambia el resultado esperado
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        with patch("xgboost.XGBClassifier") as mock_model:
            mock_model.return_value.predict.return_value = np.array([0])

            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)  # Cambia el código de respuesta esperado
        self.assertEqual(response.json(), {"detail": "Unknown column found"})  # Cambia el resultado esperado