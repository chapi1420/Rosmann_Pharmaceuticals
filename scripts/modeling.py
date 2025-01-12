import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
import logging

logging.basicConfig(
    filename="store_sales_prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SalesPrediction:
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.train_data = None
        self.test_data = None
        self.store_data = None
        self.model_pipeline = None
        self.scaler = MinMaxScaler()
        logging.info("StoreSalesPrediction class initialized.")

    def load_and_merge_data(self):
        logging.info("Loading data...")
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        self.store_data = pd.read_csv(self.store_path)

        logging.info("Merging train and test data with store data...")
        self.train_data = self.train_data.merge(self.store_data, on="Store", how="left")
        self.test_data = self.test_data.merge(self.store_data, on="Store", how="left")

    def preprocess_data(self, data):
        logging.info("Preprocessing data...")
        data['Date'] = pd.to_datetime(data['Date'])
        data['Weekday'] = data['Date'].dt.weekday
        data['IsWeekend'] = data['Weekday'].isin([5, 6]).astype(int)
        data['Month'] = data['Date'].dt.month
        data['Year'] = data['Date'].dt.year
        data['MonthPhase'] = data['Date'].dt.day.apply(self._month_phase)

        data.fillna({
            'CompetitionDistance': data['CompetitionDistance'].median(),
            'Promo2SinceWeek': 0,
            'PromoInterval': 'Unknown'
        }, inplace=True)

        return data

    def _month_phase(self, day):
        if day <= 10:
            return 'Beginning'
        elif day <= 20:
            return 'Middle'
        else:
            return 'End'

    def prepare_features(self):
        logging.info("Preparing features...")
        self.train_data = self.preprocess_data(self.train_data)
        self.test_data = self.preprocess_data(self.test_data)

    def build_pipeline(self):
        logging.info("Building pipeline...")
        numeric_features = ['CompetitionDistance', 'Promo', 'Weekday']
        categorical_features = ['StoreType', 'Assortment', 'MonthPhase']

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def train_model(self):
        logging.info("Training model...")
        X = self.train_data.drop(columns=['Sales', 'Date'])
        y = self.train_data['Sales']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        self.model_pipeline.fit(X_train, y_train)

        y_pred = self.model_pipeline.predict(X_val)
        logging.info(f"Validation Results - MAE: {mean_absolute_error(y_val, y_pred)}, MSE: {mean_squared_error(y_val, y_pred)}")

    def save_model(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"sales_model_{timestamp}.pkl"
        joblib.dump(self.model_pipeline, filename)
        logging.info(f"Model saved as {filename}")

    def train_lstm(self):
        logging.info("Training LSTM model...")

        # Prepare time series data
        data = self.train_data[['Date', 'Sales']].copy()
        data.set_index('Date', inplace=True)
        data = data.sort_index()

        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(30, len(scaled_data)):
            X.append(scaled_data[i-30:i])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping]
        )

        # Evaluate model
        val_loss = model.evaluate(X_val, y_val)
        logging.info(f"LSTM Validation Loss: {val_loss}")

        # Save model
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model.save(f"lstm_sales_model_{timestamp}.h5")
        logging.info(f"LSTM model saved as lstm_sales_model_{timestamp}.h5")