# Stock prediction pipeline

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import mlflow.sklearn

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Calculate additional features
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['VolumeChange'] = df['Volume'].pct_change()
    df['HighLowDiff'] = df['High'] - df['Low']

    # Drop NaN values
    df.dropna(inplace=True)

    # Prepare features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20', 'VolumeChange', 'HighLowDiff']
    X = df[features]
    y = df['Close'].shift(-1)  # Predict next day's closing price

    # Drop last row as it won't have a target value
    X = X[:-1]
    y = y[:-1]

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_xgboost(X_train, y_train):
    """Train an XGBoost model."""
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    """Train a LightGBM model."""
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_lstm_model(input_shape):
    """Create an LSTM model."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_train, y_train):
    """Train an LSTM model."""
    # Reshape input for LSTM [samples, time steps, features]
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = create_lstm_model((1, X_train.shape[1]))
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    if isinstance(model, Sequential):  # LSTM model
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        predictions = model.predict(X_test_reshaped).flatten()
    else:
        predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return rmse, r2

def main():
    # Fetch data
    df = fetch_stock_data('MASB.JK', '2020-01-01', '2023-12-31')

    # Preprocess data
    X, y, scaler = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run():
        # Train and evaluate XGBoost
        xgb_model = train_xgboost(X_train, y_train)
        xgb_rmse, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)

        # Log XGBoost metrics
        mlflow.log_metric("XGBoost_RMSE", xgb_rmse)
        mlflow.log_metric("XGBoost_R2", xgb_r2)

        # Log XGBoost model
        mlflow.sklearn.log_model(xgb_model, "xgboost_model")

        # Train and evaluate other models
        rf_model = train_random_forest(X_train, y_train)
        rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)

        lgb_model = train_lightgbm(X_train, y_train)
        lgb_rmse, lgb_r2 = evaluate_model(lgb_model, X_test, y_test)

        lstm_model = train_lstm(X_train, y_train)
        lstm_rmse, lstm_r2 = evaluate_model(lstm_model, X_test, y_test)

        # Log other model metrics
        mlflow.log_metric("RandomForest_RMSE", rf_rmse)
        mlflow.log_metric("RandomForest_R2", rf_r2)
        mlflow.log_metric("LightGBM_RMSE", lgb_rmse)
        mlflow.log_metric("LightGBM_R2", lgb_r2)
        mlflow.log_metric("LSTM_RMSE", lstm_rmse)
        mlflow.log_metric("LSTM_R2", lstm_r2)

        # Print results
        print(f"XGBoost - RMSE: {xgb_rmse:.4f}, R2: {xgb_r2:.4f}")
        print(f"Random Forest - RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")
        print(f"LightGBM - RMSE: {lgb_rmse:.4f}, R2: {lgb_r2:.4f}")
        print(f"LSTM - RMSE: {lstm_rmse:.4f}, R2: {lstm_r2:.4f}")

if __name__ == "__main__":
    main()
