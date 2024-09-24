# Azure ML training script

import os
import argparse
import pandas as pd
import numpy as np
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import joblib
import yfinance as yf

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

def train_model(X_train, y_train):
    """Train an XGBoost model."""
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Get the experiment run context
    run = Run.get_context()

    # Get input dataset by name
    df = fetch_stock_data('MASB.JK', '2020-01-01', '2023-12-31')

    # Preprocess data
    X, y, scaler = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    run.log('MSE', mse)
    run.log('RMSE', rmse)
    run.log('R2', r2)

    # If using MLflow
    mlflow.log_metric('MSE', mse)
    mlflow.log_metric('RMSE', rmse)
    mlflow.log_metric('R2', r2)

    # Save the model
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/stock_prediction_model.pkl'
    joblib.dump(model, model_path)

    # Upload the model file explicitly into artifacts
    run.upload_file(name=model_path, path_or_stream=model_path)

    # Register the model
    run.register_model(model_name='stock_prediction_model', model_path=model_path)

    print(f'Model trained and registered. MSE: {mse}, RMSE: {rmse}, R2: {r2}')

if __name__ == "__main__":
    main()
