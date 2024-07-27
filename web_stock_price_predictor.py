import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# Input for stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define the date range
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

try:
    # Download stock data
    google_data = yf.download(stock, start, end)
    st.subheader("Stock Data")
    st.write(google_data)
except Exception as e:
    st.error(f"Error downloading stock data: {e}")

try:
    model = load_model("Latest_stock_price_model.keras")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

try:
    st.subheader('Original Close Price and MA for 250 days')
    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))
except Exception as e:
    st.error(f"Error plotting 250-day MA: {e}")

try:
    st.subheader('Original Close Price and MA for 200 days')
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))
except Exception as e:
    st.error(f"Error plotting 200-day MA: {e}")

try:
    st.subheader('Original Close Price and MA for 100 days')
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))
except Exception as e:
    st.error(f"Error plotting 100-day MA: {e}")

try:
    st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))
except Exception as e:
    st.error(f"Error plotting 100-day and 250-day MA: {e}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

try:
    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    plotting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=google_data.index[splitting_len + 100:]
    )
    st.subheader("Original values vs Predicted values")
    st.write(plotting_data)
except Exception as e:
    st.error(f"Error generating predictions: {e}")

try:
    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data.Close[:splitting_len + 100], plotting_data], axis=0))
    plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error plotting original vs predicted close price: {e}")
