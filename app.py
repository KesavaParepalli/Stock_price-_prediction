# from flask import Flask, render_template, request
# from utils import predict_next_price

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     ticker = ""
#     if request.method == 'POST':
#         ticker = request.form['ticker'].upper()
#         try:
#             prediction = predict_next_price(ticker)
#         except Exception as e:
#             prediction = f"Error: {str(e)}"
#     return render_template('index.html', prediction=prediction, ticker=ticker)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model and scaler
model = load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Dropdown options (Add more if needed)
tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Infosys (INFY)": "INFY.NS",
    "TCS (TCS)": "TCS.NS",
    "Reliance (RELIANCE)": "RELIANCE.NS"
}

def get_recent_data(ticker):
    df = yf.download(ticker, period="90d", progress=False)
    if df.empty or 'Close' not in df.columns:
        raise ValueError("No data found for ticker.")
    
    close_prices = df['Close'].dropna().values[-60:]
    if len(close_prices) < 60:
        raise ValueError("Not enough data for prediction.")

    scaled_data = scaler.transform(close_prices.reshape(-1, 1))
    return scaled_data, close_prices

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_url = None
    selected_ticker = None

    if request.method == 'POST':
        selected_ticker = request.form.get('ticker')
        try:
            scaled_data, raw_prices = get_recent_data(selected_ticker)
            input_data = scaled_data.reshape(1, 60, 1)

            predicted_scaled = model.predict(input_data)[0][0]
            predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
            prediction = round(predicted_price, 2)

            # Plot the prices
            plt.figure(figsize=(10, 5))
            plt.plot(raw_prices, label='Last 60 Days Closing Price', marker='o')
            plt.title(f'{selected_ticker} Stock Price')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()

            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)
            plt.close()
            plot_url = plot_path

        except Exception as e:
            print("Error:", e)
            prediction = "Error fetching data or predicting."
            plot_url = None

    return render_template('index.html', prediction=prediction, plot_url=plot_url, tickers=tickers, selected_ticker=selected_ticker)

if __name__ == '__main__':
    app.run(debug=True)
