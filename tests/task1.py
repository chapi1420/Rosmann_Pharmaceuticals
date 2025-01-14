import unittest
import pandas as pd
import os

class TestDataExplorer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create dummy CSV files for testing
        cls.train_data = pd.DataFrame({
            'Store': [1, 2, 3],
            'Sales': [200, 300, 400],
            'Customers': [20, 30, 40],
            'Promo': [1, 0, 1],
            'StateHoliday': ['0', 'a', '0'],
            'Assortment': ['a', 'b', 'c'],
            'CompetitionDistance': [1000, 2000, 1500]
        })
        cls.test_data = pd.DataFrame({
            'Store': [1, 2],
            'Promo': [1, 0],
            'StateHoliday': ['0', '0'],
        })
        cls.store_data = pd.DataFrame({
            'Store': [1, 2, 3],
            'Assortment': ['a', 'b', 'c'],
            'CompetitionDistance': [1000, 2000, 1500]
        })
        
        cls.train_data.to_csv('train.csv', index=False)
        cls.test_data.to_csv('test.csv', index=False)
        cls.store_data.to_csv('store.csv', index=False)

        # Initialize DataExplorer
        cls.explorer = DataExplorer('train.csv', 'test.csv', 'store.csv')

    @classmethod
    def tearDownClass(cls):
        # Remove the dummy CSV files after tests
        os.remove('train.csv')
        os.remove('test.csv')
        os.remove('store.csv')

    def test_load_data(self):
        self.explorer.load_data()
        self.assertIsNotNone(self.explorer.train_data)
        self.assertIsNotNone(self.explorer.test_data)
        self.assertIsNotNone(self.explorer.store_data)
        self.assertEqual(len(self.explorer.train_data), 3)
        self.assertEqual(len(self.explorer.test_data), 2)

    def test_merge_data(self):
        self.explorer.load_data()
        self.explorer.merge_data()
        self.assertIn('Assortment', self.explorer.train_data.columns)
        self.assertIn('CompetitionDistance', self.explorer.test_data.columns)

    def test_clean_data(self):
        self.explorer.load_data()
        self.explorer.merge_data()
        self.explorer.clean_data()
        self.assertEqual(self.explorer.train_data['Sales'].isnull().sum(), 0)
        self.assertEqual(self.explorer.test_data['Promo'].isnull().sum(), 0)

    def test_analyze_data(self):
        self.explorer.load_data()
        self.explorer.merge_data()
        self.explorer.clean_data()
        self.explorer.analyze_data()
        # Check if the logger has logged the analysis steps
        with open('task1_exploration.log', 'r') as log_file:
            logs = log_file.read()
            self.assertIn("Performing exploratory analysis", logs)

if __name__ == "__main__":
    unittest.main()
