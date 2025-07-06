import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
import time
import requests
from alpha_vantage.timeseries import TimeSeries

# Set page style
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f9fa;
            color: #495057;
        }
        .title {
            font-size: 2em;
            color: #3171e0;
            text-align: center;
            padding: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Set date ranges
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')  # Last trading day (July 04, 2025)
start_training = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')  # 3 years ago (July 06, 2022)
start_display = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')  # Last 30 days

# Authentication functions
def make_account():
    email = st.text_input("Enter your email:", key="signup_email")
    username = st.text_input("Enter a username:", key="signup_username")
    password = st.text_input("Enter a password:", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm password:", type="password", key="signup_confirm_password")
    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("❌ Passwords do not match.")
            return
        if not email or not username or not password:
            st.error("❌ All fields are required.")
            return
        try:
            with open("email.txt", "a") as f:
                f.write(f"{email}\n")
            with open(f"{username}_username.txt", "w") as f:
                f.write(username)
            with open(f"{username}_password.txt", "w") as f:
                f.write(password)
            with open(f"{username}_email.txt", "w") as f:
                f.write(email)
            st.success("✅ Account created successfully!")
        except Exception as e:
            st.error(f"❌ Error creating account: {e}")

def login():
    username = st.text_input("Enter your username:", key="login_username")
    password = st.text_input("Enter your password:", type="password", key="login_password")
    if st.button("Log In"):
        check(username, password)

def check(username, password):
    username_file_path = f"{username}_username.txt"
    password_file_path = f"{username}_password.txt"
    if os.path.exists(username_file_path) and os.path.exists(password_file_path):
        stored_username = open(username_file_path).read().strip()
        stored_password = open(password_file_path).read().strip()
        if username == stored_username and password == stored_password:
            st.success("✅ Successful login")
            st.session_state.authenticated = True
        else:
            st.error('❌ Incorrect username or password')
    else:
        st.error("❌ Account not found. Please sign up.")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

st.sidebar.title("🔐 Login / Sign Up")
login()
st.sidebar.markdown("---")
make_account()

# Main app if authenticated
if st.session_state.authenticated:
    st.title("📈 Stock Price Viewer")

    ticker = st.text_input(
        'Enter Stock Ticker Symbol (e.g., AAPL for Apple, MSFT for Microsoft)',
        'AAPL'
    ).upper()

    # Check internet connection
    def check_internet():
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except requests.ConnectionError:
            return False

    # Download data with Alpha Vantage
    def fetch_data(ticker, start, end, api_key, max_retries=5):
        if not check_internet():
            st.error("❌ No internet connection. Please check your network.")
            return None
        for attempt in range(max_retries):
            try:
                ts = TimeSeries(key=api_key, output_format='pandas')
                df, meta = ts.get_daily(symbol=ticker, outputsize='full')
                df.index = pd.to_datetime(df.index)
                df = df[start:end].sort_index()
                if df.empty:
                    st.warning(f"Attempt {attempt + 1}/{max_retries}: No data in range {start} to {end}. Retrying with full history...")
                    df = df.sort_index()  # Fallback to all available data
                if not df.empty:
                    st.success(f"✅ Data loaded for {ticker} from {start} to {end}")
                    return df
                time.sleep(2 ** attempt)
            except Exception as e:
                st.warning(f"Attempt {attempt + 1}/{max_retries}: Error - {e}. Retrying...")
                time.sleep(2 ** attempt)
        st.error(f"❌ Failed to fetch data after {max_retries} attempts. Verify API key, ticker ({ticker}), or try again later.")
        return None

    # Replace with your own API key
    API_KEY = "YOUR_API_KEY_HERE"  # Obtain from https://www.alphavantage.co/support/#api-key

    df = fetch_data(ticker, start_training, end, API_KEY)
    if df is not None and '4. close' in df.columns:
        df['Close'] = df['4. close']
    else:
        df = None
        st.error("❌ No data or invalid format. Check API key, ticker, or internet.")

    if df is not None:
        st.success(f"✅ Data loaded for {ticker} from {start_training} to {end}")

        # Filter for last 30 days for display
        df_display = df[start_display:end]

        # Data summary
        st.subheader(f'Data Summary (Last 30 Days)')
        st.write(df_display.describe())

        # Closing price plot (last 30 days)
        st.subheader('📉 Closing Price Over Time (Last 30 Days)')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df_display.index, df_display['Close'], label='Closing Price', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title(f'{ticker} Closing Price (Last 30 Days)')
        plt.legend()
        st.pyplot(fig)

        # LSTM prediction for future days based on 3 years of data
        st.subheader('🔮 Stock Price Prediction (Next N Days)')
        future_days = st.slider('Select number of future days to predict', min_value=1, max_value=7, value=3)  # Limit to 7 days

        # Prepare data (use all 3 years for training)
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Use all 3 years as training data
        train_data = scaled_data

        # Create sequences for training
        sequence_length = 30  # Adjusted to fit within 3 years, using last 30 days as sequence
        x_train, y_train = [], []
        if len(train_data) > sequence_length:
            for i in range(sequence_length, len(train_data)):
                x_train.append(train_data[i-sequence_length:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        else:
            st.error("❌ Insufficient data for training. Need at least 30 days.")
            x_train, y_train = np.array([]), np.array([])

        # Load or train model
        model_file_path = 'ker_model_3y.h5'  # Unique filename for 3-year model
        model = None
        if os.path.exists(model_file_path):
            try:
                model = load_model(model_file_path)
                st.success("✅ Loaded existing model for prediction.")
            except Exception as e:
                st.warning(f"⚠️ Error loading model: {e}. Training a new model...")
        else:
            st.warning("⚠️ No pre-trained model found. Training a new model...")

        if model is None and len(x_train) > 0:
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=False, input_shape=(sequence_length, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)  # Reduced epochs for speed
            model.save(model_file_path)
            st.success("✅ New model trained and saved.")

        # Future prediction
        if model is not None and len(x_train) > 0:
            last_sequence = scaled_data[-sequence_length:]  # Use last 30 days from 3 years
            future_predictions = []

            current_input = last_sequence.reshape((1, sequence_length, 1))
            for _ in range(future_days):
                next_pred_scaled = model.predict(current_input, verbose=0)
                future_predictions.append(next_pred_scaled[0, 0])
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_pred_scaled[0, 0]

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions_rescaled = scaler.inverse_transform(future_predictions).flatten()

            last_date = df_display.index[-1]  # Start from the last 30-day point
            future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

            st.subheader(f'Next {future_days} Days Forecast for {ticker} (Based on 3 Years)')
            fig4 = plt.figure(figsize=(12, 6))
            plt.plot(df_display.index, df_display['Close'], label='Historical Price (Last 30 Days)', color='blue')
            plt.plot(future_dates, future_predictions_rescaled, label='Forecast', color='red', linestyle='--', marker='o')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'{ticker} Closing Price Forecast')
            plt.legend()
            st.pyplot(fig4)
        else:
            st.warning("⚠️ Prediction skipped due to insufficient data or model issues.")

        # Feedback section
        st.subheader('💬 User Feedback')
        st.markdown("Provide your feedback below. Let us know how we can improve! 🚀")
        user_feedback = st.text_area('Feedback', value='', placeholder='Type your feedback here...')
        if st.button('Submit Feedback'):
            if user_feedback.strip():
                feedback_folder = 'user_feedback'
                os.makedirs(feedback_folder, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                feedback_file_path = os.path.join(feedback_folder, f'feedback_{timestamp}.txt')
                with open(feedback_file_path, 'w') as feedback_file:
                    feedback_file.write(user_feedback)
                st.success('✅ Feedback submitted! Thank you! 🙌')
            else:
                st.warning('⚠️ Please provide meaningful feedback.')

else:
    st.warning("🔐 Please log in to access the app.")