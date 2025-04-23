from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Initialize the Flask app
app = Flask(__name__)

# Step 1: Input stock tickers
def preprocess(series):
    sequence_length = 60  # 60-day input sequence
    prediction_horizon = 10  # 10-day prediction horizon

    series = series.dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    X, y = [], []

    for i in range(len(scaled_series) - sequence_length - prediction_horizon):
        X.append(scaled_series[i:i + sequence_length])
        y.append(scaled_series[i + sequence_length + prediction_horizon - 1])

    return np.array(X), np.array(y), scaler

def predict_returns(tickers):
    results = []
    user_threshold = 15  # percent return target

    end = datetime.today()
    start = end - timedelta(days=365 * 10)

    for ticker in tickers:
        print(f"Training model for {ticker}...")
        data = yf.download(ticker, start=start, end=end, interval='1d')['Close']
        X, y, scaler = preprocess(data)

        if len(X) == 0:
            print(f"Not enough data for {ticker}. Skipping...")
            continue

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.3),
            LSTM(64),
            Dense(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)

        # Predict future return
        latest_seq = data.dropna().values[-60:].reshape(-1, 1)
        latest_scaled = scaler.transform(latest_seq)
        predicted_scaled = model.predict(latest_scaled.reshape(1, 60, 1))[0, 0]
        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0, 0]
        current_price = latest_seq[-1][0]
        predicted_return = (predicted_price - current_price) / current_price * 100

        if predicted_return >= user_threshold:
            results.append({
                'Ticker': ticker,
                'Current_Price': current_price,
                'Predicted_Price': predicted_price,
                'Predicted_Return_%': predicted_return
            })

    return results

# Step 2: Set up Flask route to handle form submission
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tickers_input = request.form["tickers"]
        tickers = [ticker.strip() + ".NS" for ticker in tickers_input.split(',')]
        
        # Call the function to predict returns
        results = predict_returns(tickers)
        
        # Return the results to the HTML page
        return render_template("index.html", results=results)
    
    return render_template("index.html", results=None)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
