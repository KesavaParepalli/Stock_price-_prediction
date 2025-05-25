import numpy as np
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model

def fetch_data(ticker, period='90d'):
    df = yf.download(ticker, period=period)
    return df['Close'].values.reshape(-1, 1)

def load_scaler_and_model():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model = load_model('model.h5')
    return scaler, model

def predict_next_price(ticker):
    data = fetch_data(ticker)
    scaler, model = load_scaler_and_model()

    scaled_data = scaler.transform(data)
    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    pred_scaled = model.predict(last_60)
    pred = scaler.inverse_transform(pred_scaled)
    return round(pred[0][0], 2)
