import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

class DataExplorer:
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.train_data = None
        self.test_data = None
        self.store_data = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            filename='task1_exploration.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_path, low_memory=False)
            self.test_data = pd.read_csv(self.test_path, low_memory=False)
            self.store_data = pd.read_csv(self.store_path, low_memory=False)
            self.logger.info("Datasets loaded successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Error loading datasets: {e}")
            raise

    def merge_data(self):
        self.logger.info("Merging datasets")
        self.train_data = self.train_data.merge(self.store_data, on='Store', how='left')
        self.test_data = self.test_data.merge(self.store_data, on='Store', how='left')
        self.logger.info("Datasets merged successfully")

    def clean_data(self):
        self.logger.info("Cleaning data")

        # Handle missing values
        missing_cols = self.train_data.columns[self.train_data.isnull().any()]
        self.logger.info(f"Columns with missing values: {missing_cols}")

        imputer = SimpleImputer(strategy='median')
        self.train_data[missing_cols] = imputer.fit_transform(self.train_data[missing_cols])
        self.test_data[missing_cols] = imputer.transform(self.test_data[missing_cols])
        self.logger.info("Missing values imputed")

        # Handle outliers
        numeric_cols = self.train_data.select_dtypes(include=np.number).columns
        z_scores = np.abs(zscore(self.train_data[numeric_cols]))
        self.train_data = self.train_data[(z_scores < 3).all(axis=1)]
        self.logger.info("Outliers handled")

    def analyze_data(self):
        self.logger.info("Performing exploratory analysis")
        self._plot_promo_distribution()
        self._plot_sales_behavior_holidays()
        self._analyze_correlation()
        self._plot_promo_effect()
        self._plot_assortment_effect()
        self._plot_competition_effect()

    def _plot_promo_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Promo', data=self.train_data)
        plt.title("Promotion Distribution in Training Set")
        plt.savefig("promo_distribution_train.png")
        plt.close()

        sns.countplot(x='Promo', data=self.test_data)
        plt.title("Promotion Distribution in Testing Set")
        plt.savefig("promo_distribution_test.png")
        plt.close()
        self.logger.info("Saved promotion distribution plots")

    def _plot_sales_behavior_holidays(self):
        state_holidays = self.train_data[self.train_data['StateHoliday'] != '0']
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='StateHoliday', y='Sales', data=state_holidays)
        plt.title("Sales Behavior During Holidays")
        plt.savefig("sales_behavior_holidays.png")
        plt.close()
        self.logger.info("Saved sales behavior during holidays plot")

    def _analyze_correlation(self):
        correlation = self.train_data[['Sales', 'Customers']].corr()
        self.logger.info(f"Correlation between Sales and Customers: {correlation.loc['Sales', 'Customers']}")

    def _plot_promo_effect(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Promo', y='Sales', data=self.train_data)
        plt.title("Promo Effect on Sales")
        plt.savefig("promo_effect_sales.png")
        plt.close()
        self.logger.info("Saved promo effect on sales plot")

    def _plot_assortment_effect(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Assortment', y='Sales', data=self.train_data)
        plt.title("Assortment Type Effect on Sales")
        plt.savefig("assortment_effect_sales.png")
        plt.close()
        self.logger.info("Saved assortment type effect on sales plot")

    def _plot_competition_effect(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='CompetitionDistance', y='Sales', data=self.train_data)
        plt.title("Competition Distance Effect on Sales")
        plt.savefig("competition_distance_sales.png")
        plt.close()
        self.logger.info("Saved competition distance effect on sales plot")

if __name__ == "__main__":
    explorer = DataExplorer(
        train_path="C:\\Users\\nadew\\10x\\week4\\technical doc\\Data-20250101T153622Z-001\\Data\\rossmann-store-sales\\store.csv", 
        test_path="C:\\Users\\nadew\\10x\\week4\\technical doc\\Data-20250101T153622Z-001\\Data\\rossmann-store-sales\\test.csv", 
        store_path= "C:\\Users\\nadew\\10x\\week4\\technical doc\\Data-20250101T153622Z-001\\Data\\rossmann-store-sales\\train.csv"
    )
    explorer.load_data()
    explorer.merge_data()
    explorer.clean_data()
    explorer.analyze_data()
    explorer.logger.info("Task 1 completed successfully")
