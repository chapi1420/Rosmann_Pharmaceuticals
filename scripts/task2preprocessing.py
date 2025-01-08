import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

class DataProcessor:
    def __init__(self, train_file, test_file, store_file):
        self.train_file = train_file
        self.test_file = test_file
        self.store_file = store_file
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def load_data(self):
        self.train_df = pd.read_csv(self.train_file, low_memory=False)
        self.test_df = pd.read_csv(self.test_file, low_memory=False)
        self.store_df = pd.read_csv(self.store_file, low_memory=False)


        logging.info(f"Train columns: {list(self.train_df.columns)}")
        logging.info(f"Test columns: {list(self.test_df.columns)}")
        logging.info(f"Store columns: {list(self.store_df.columns)}")


    def preprocess_data(self):
        # Merge train_df and store_df
        self.data = pd.merge(self.train_df, self.store_df, on='Store', how='left')

        # Split train_df into train and test sets
        self.X = self.data.drop(['Sales'], axis=1)
        self.y = self.data['Sales']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        required_columns = ['Date', 'StateHoliday', 'Assortment', 'StoreType', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        for col in required_columns:
            if col not in self.X.columns:
                logging.warning(f"'{col}' column not found in self.X. Skipping processing for this column.")
            else:
                if col == 'Date':
                    self.X_train['Date'] = pd.to_datetime(self.X_train['Date'])
                    self.X_test['Date'] = pd.to_datetime(self.X_test['Date'])
                    self.X_train['CompetitionOpen'] = (self.X_train['Date'].dt.year - self.X_train['CompetitionOpenSinceYear']) * 12 + (self.X_train['Date'].dt.month - self.X_train['CompetitionOpenSinceMonth'])
                    self.X_test['CompetitionOpen'] = (self.X_test['Date'].dt.year - self.X_test['CompetitionOpenSinceYear']) * 12 + (self.X_test['Date'].dt.month - self.X_test['CompetitionOpenSinceMonth'])
                    self.X_train['CompetitionOpen'] = self.X_train['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
                    self.X_test['CompetitionOpen'] = self.X_test['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
                elif col == 'StateHoliday':
                    self.X_train['StateHoliday'] = self.X_train['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3})
                    self.X_test['StateHoliday'] = self.X_test['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3})
                elif col == 'Assortment':
                    self.X_train['Assortment'] = self.X_train['Assortment'].replace({'a': 1, 'b': 2, 'c': 3})
                    self.X_test['Assortment'] = self.X_test['Assortment'].replace({'a': 1, 'b': 2, 'c': 3})
                elif col == 'StoreType':
                    self.X_train['StoreType'] = self.X_train['StoreType'].replace({'a': 1, 'b': 2, 'c': 3, 'd': 4})
                    self.X_test['StoreType'] = self.X_test['StoreType'].replace({'a': 1, 'b': 2, 'c': 3, 'd': 4})
                else:
                    self.X_train[col] = self.X_train[col].fillna(0)
                    self.X_test[col] = self.X_test[col].fillna(0)

        columns_to_drop = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        self.X_train = self.X_train.drop(columns=[col for col in columns_to_drop if col in self.X_train.columns], axis=1)
        self.X_test = self.X_test.drop(columns=[col for col in columns_to_drop if col in self.X_test.columns], axis=1)
        self.X_test['sales'] = np.nan
        
        #save
        self.train_df.to_csv('C:\\Users\\nadew\\10x\\week4\\Rosmann\\rossmann-store-sales\\processed_train_data.csv', index=False)
        self.test_df.to_csv('C:\\Users\\nadew\\10x\\week4\\Rosmann\\rossmann-store-sales\\processed_test_data.csv', index=False)



    def split_data(self):
        self.train_data = self.data[self.data['Sales'].notnull()]
        self.test_data = self.data[self.data['Sales'].isnull()]

        logging.info(f"Train data shape: {self.train_data.shape}")
        logging.info(f"Test data shape: {self.test_data.shape}")

    def get_data(self):
        return self.train_data, self.test_data

# if __:
#     data_processor = DataProcessor('train.csv', 'test.csv', 'store.csv')
#     data_processor.load_data()
#     data_processor.preprocess_data()
#     data_processor.split_data()
#     train_data, test_data = data_processor.get_data()
