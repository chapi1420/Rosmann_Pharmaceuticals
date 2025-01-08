import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import logging
import joblib

class SalesPredictor:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def train_model(self):
        # Initialize the Random Forest Regressor
        self.model = RandomForestRegressor(random_state=42)

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
                                   scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)

        # Fit the model
        grid_search.fit(self.X_train, self.y_train)

        # Get the best model
        self.model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")

    def evaluate_model(self):
        # Make predictions
        predictions = self.model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, predictions)
        logging.info(f"Mean Squared Error: {mse}")

        return predictions

    def save_model(self):
        model_filename = f"sales_predictor_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        joblib.dump(self.model, model_filename)
        logging.info(f"Model saved as: {model_filename}")

# if __name__ == "__main__":
#     # Assuming train_data and test_data are obtained from DataProcessor
#     train_data, test_data = data_processor.get_data()
    
#     # Define X and y
#     X_train = train_data.drop(['Sales'], axis=1)
#     y_train = train_data['Sales']
#     X_test = test_data.drop(['Sales'], axis=1)  # Ensure test data also has the same features
#     y_test = test_data['Sales'] if 'Sales' in test_data.columns else None  # Only if available

#     sales_predictor = SalesPredictor(X_train, y_train, X_test, y_test)
#     sales_predictor.train_model()
#     predictions = sales_predictor.evaluate_model()
#     sales_predictor.save_model()
