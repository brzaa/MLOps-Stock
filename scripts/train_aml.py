import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import yfinance as yf
import os

def fetch_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

def preprocess_data(df):
    df['Returns'] = df['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20']
    X = df[features]
    y = df['Close'].shift(-1)[:-1]  # Predict next day's closing price
    X = X[:-1]  # Remove last row to align with y

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    df = fetch_stock_data('MASB.JK', '2020-01-01', '2023-12-31')
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')

    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/stock_prediction_model.pkl'
    joblib.dump(model, model_path)

    print(f'Model trained and saved to {model_path}')

if __name__ == "__main__":
    main()
