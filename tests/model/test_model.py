import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xgboost import XGBClassifier  # Importar el XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.xgb_model = XGBClassifier(random_state=1, learning_rate=0.01)
        self.model = DelayModel(model=self.xgb_model)

        github_workspace = os.environ['GITHUB_WORKSPACE']
        data_file_path = os.path.join(github_workspace, 'data', 'data.csv')
        self.data = pd.read_csv(filepath_or_buffer=data_file_path)

    def test_model_preprocess_for_training(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        features = features[self.FEATURES_COLS]

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[1], len(self.FEATURES_COLS))
        self.assertEqual(set(features.columns), set(self.FEATURES_COLS))

        self.assertIsInstance(target, pd.Series)
        self.assertEqual(target.name, "delay")

    def test_model_preprocess_for_serving(self):
        features = self.model.preprocess(data=self.data)
        features = features[self.FEATURES_COLS]

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[1], len(self.FEATURES_COLS))
        self.assertEqual(set(features.columns), set(self.FEATURES_COLS))

    def test_model_fit(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        features = features[self.FEATURES_COLS]
        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        ### Data Balance
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1
        
        self.model._model.set_params(scale_pos_weight=scale)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True) 
        
        self.assertLess(report["0"]["recall"], 0.60) # Para verificar si esta correctamente identificando las clases 0
        self.assertLess(report["0"]["f1-score"], 0.70)
        self.assertGreater(report["1"]["recall"], 0.60) # Para verificar si esta correctamente identificando las clases 1
        self.assertGreater(report["1"]["f1-score"], 0.30)

        # f1-score = (2×Precision×Recall)/(Precision+Recall)
        """ 
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30 """

    def test_model_predict(self):
        #features = self.model.preprocess(data=self.data)
        #features = features[self.FEATURES_COLS]
        features = self.data
        
        #self.model.fit(features=features, target=target)
        predicted_targets = self.model.predict(features=features)

        self.assertIsInstance(predicted_targets, list)
        self.assertEqual(len(predicted_targets), features.shape[0])
        self.assertTrue(all(isinstance(predicted_target, int) for predicted_target in predicted_targets))