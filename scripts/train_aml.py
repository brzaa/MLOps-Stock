# Azure ML training script
import os
import pandas as pd
import numpy as np
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yfinance as yf

# Try to import Azure ML Run, but don't fail if it's not available
try:
    from azureml.core import Run
    run = Run.get_context()
except ImportError:
    run = None
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    df = yf.download(symbol, start=start_date, end=end_date)
    return model

def main():
    # Get the experiment run context
    run = Run.get_context()
    # Get input dataset by name
    df = fetch_stock_data('MSFT', '2020-01-01', '2023-12-31')
    df = fetch_stock_data('MASB.JK', '2020-01-01', '2023-12-31')

    # Preprocess data
    X, y, scaler = preprocess_data(df)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    run.log('MSE', mse)
    run.log('RMSE', rmse)
    run.log('R2', r2)
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')
    if run is not None and hasattr(run, 'log'):
        run.log('MSE', mse)
        run.log('RMSE', rmse)
        run.log('R2', r2)

    # If using MLflow
    # Log metrics with MLflow
    mlflow.log_metric('MSE', mse)
    mlflow.log_metric('RMSE', rmse)
    mlflow.log_metric('R2', r2)
    model_path = 'outputs/stock_prediction_model.pkl'
    joblib.dump(model, model_path)

    # Upload the model file explicitly into artifacts
    run.upload_file(name=model_path, path_or_stream=model_path)
    # Log the model with MLflow
    mlflow.sklearn.log_model(model, "model")
    if run is not None and hasattr(run, 'upload_file'):
        run.upload_file(name=model_path, path_or_stream=model_path)

    # Register the model
    run.register_model(model_name='stock_prediction_model', model_path=model_path)
    if run is not None and hasattr(run, 'register_model'):
        run.register_model(model_name='stock_prediction_model', model_path=model_path)

    print(f'Model trained and registered. MSE: {mse}, RMSE: {rmse}, R2: {r2}')
    print(f'Model trained and saved. MSE: {mse}, RMSE: {rmse}, R2: {r2}')

if __name__ == "__main__":
    main()
