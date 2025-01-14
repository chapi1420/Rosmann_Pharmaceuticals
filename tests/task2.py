import unittest
import pandas as pd
import os

class TestSalesPrediction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create dummy CSV files for testing
        cls.train_data = pd.DataFrame({
            'Store': [1, 2, 3],
            'Sales': [200, 300, 400],
            'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'Promo': [1, 0, 1]
        })
        cls.test_data = pd.DataFrame({
            'Store': [1, 2],
            'Date': ['2022-01-04', '2022-01-05'],
            'Promo': [1, 0],
        })
        cls.store_data = pd.DataFrame({
            'Store': [1, 2, 3],
            'StoreType': ['a', 'b', 'c'],
            'Assortment': ['a', 'b', 'c'],
            'CompetitionDistance': [1000, 2000, 1500]
        })

        cls.train_data.to_csv('train.csv', index=False)
        cls.test_data.to_csv('test.csv', index=False)
        cls.store_data.to_csv('store.csv', index=False)

        # Initialize SalesPrediction
        cls.sales_prediction = SalesPrediction('train.csv', 'test.csv', 'store.csv')

    @classmethod
    def tearDownClass(cls):
        # Remove the dummy CSV files after tests
        os.remove('train.csv')
        os.remove('test.csv')
        os.remove('store.csv')

    def test_load_and_merge_data(self):
        self.sales_prediction.load_and_merge_data()
        self.assertIsNotNone(self.sales_prediction.train_data)
        self.assertIsNotNone(self.sales_prediction.test_data)
        self.assertEqual(len(self.sales_prediction.train_data), 3)
        self.assertEqual(len(self.sales_prediction.test_data), 2)

    def test_preprocess_data(self):
        self.sales_prediction.load_and_merge_data()
        preprocessed_train = self.sales_prediction.preprocess_data(self.sales_prediction.train_data.copy())
        self.assertIn('Weekday', preprocessed_train.columns)
        self.assertIn('IsWeekend', preprocessed_train.columns)
        self.assertEqual(preprocessed_train['CompetitionDistance'].isnull().sum(), 0)

    def test_prepare_features(self):
        self.sales_prediction.load_and_merge_data()
        self.sales_prediction.prepare_features()
        self.assertIn('Month', self.sales_prediction.train_data.columns)
        self.assertIn('Year', self.sales_prediction.train_data.columns)

    def test_build_pipeline(self):
        self.sales_prediction.build_pipeline()
        self.assertIsNotNone(self.sales_prediction.model_pipeline)

    def test_train_model(self):
        self.sales_prediction.load_and_merge_data()
        self.sales_prediction.prepare_features()
        self.sales_prediction.build_pipeline()
        self.sales_prediction.train_model()
        self.assertIsNotNone(self.sales_prediction.model_pipeline)

    def test_save_model(self):
        self.sales_prediction.load_and_merge_data()
        self.sales_prediction.prepare_features()
        self.sales_prediction.build_pipeline()
        self.sales_prediction.train_model()
        self.sales_prediction.save_model()
        # Check if the model file exists
        model_files = [f for f in os.listdir() if f.startswith("sales_model_") and f.endswith(".pkl")]
        self.assertGreater(len(model_files), 0)

    def test_train_lstm(self):
        self.sales_prediction.load_and_merge_data()
        self.sales_prediction.prepare_features()
        self.sales_prediction.train_lstm()
        # Check if the LSTM model file exists
        lstm_files = [f for f in os.listdir() if f.startswith("lstm_sales_model_") and f.endswith(".h5")]
        self.assertGreater(len(lstm_files), 0)

if __name__ == "__main__":
    unittest.main()
