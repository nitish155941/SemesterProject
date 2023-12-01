import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from googletrans import Translator
import hashlib
import base64
from datetime import timedelta


# Defining the start and end dates for the data
start = '2000-01-01'
end = '2023-11-11'

# Applying CSS styling for the Streamlit app

st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f9fa;
            color: #495057;
            margin: 0;
            padding: 0;
        }

        .title {
            font-size: 2em;
            color: #3171e0;
            text-align: center;
            padding: 1em;
        }

        .input-label {
            font-size: 1.2em;
            color: #6c757d;
            margin-top: 1em;
        }

        .user-input {
            width: 50%;
            padding: 0.5em;
            font-size: 1em;
            margin-bottom: 1em;
        }

        .subheader {
            font-size: 1.5em;
            color: #495057;
            margin-top: 1em;
        }

        .data-description {
            margin-top: 1em;
        }

        .chart {
            width: 80%;
            margin: 2em auto;
        }

        .checkbox-label {
            font-size: 1em;
            color: #6c757d;
        }

        .moving-average-plot {
            width: 80%;
            margin: 2em auto;
        }

        .closing-price-plot {
            width: 80%;
            margin: 2em auto;
        }

        .feedback-section {
            margin-top: 2em;
        }

        .feedback-label {
            font-size: 1.2em;
            color: #6c757d;
            margin-top: 1em;
        }

        .feedback-input {
            width: 70%;
            padding: 0.5em;
            font-size: 1em;
            margin-bottom: 1em;
        }

        .feedback-button {
            padding: 0.5em 1em;
            font-size: 1em;
            background-color: #28a745;
            color: #fff;
            border: none;
            cursor: pointer;
        }

        .error-message {
            color: #dc3545;
            font-size: 1.2em;
            margin-top: 1em;
        }

        .success-message {
            color: #28a745;
            font-size: 1.2em;
            margin-top: 1em;
        }
    </style>
""", unsafe_allow_html=True)
# Login and Signup for  sequrity
def make_account():
    email = st.text_input("Enter your email:")
    username = st.text_input("Enter a username:")
    password = st.text_input("Enter a password:", type="password")
    confirm_password = st.text_input("Confirm password:", type="password")
    
    if st.button("Sign Up") and password == confirm_password:
        # Save user information to email.txt
        with open("email.txt", "a") as f:
            f.write(f"{email}\n")

        # Save user information to files
        with open(f"{username}_username.txt", "w") as f:
            f.write(username)
        with open(f"{username}_password.txt", "w") as f:
            f.write(password)
        with open(f"{username}_email.txt", "w") as f:
            f.write(email)
        st.success("Account created successfully!")

def login():
    username = st.text_input("Enter your username:")
    password = st.text_input("Enter your password:", type="password")
    
    if st.button("Log In"):
        check(username, password)

def check(username, password):
    username_file_path = f"{username}_username.txt"
    password_file_path = f"{username}_password.txt"

    if os.path.exists(username_file_path) and os.path.exists(password_file_path):
        stored_username = open(username_file_path).read().strip()
        stored_password = open(password_file_path).read().strip()

        if username == stored_username and password == stored_password:
            st.success("Successful login")
            # Set authentication status in session_state
            st.session_state.authenticated = True
        else:
            st.error('Incorrect username or password')
    else:
        st.error("Account not found. Please sign up.")

# Check if the user is authenticated
if 'authenticated' not in st.session_state:
    st.sidebar.title("Login / Sign Up")
    login()
    st.sidebar.markdown("---")
    make_account()

# Check if the user is authenticated before proceeding
if 'authenticated' in st.session_state and st.session_state.authenticated:
    # Rest of your existing code here

    # Asking the user to enter the stock ticker symbol
    user_input = st.text_input('Enter Stock Ticker', 'Enter Stock Code')

    # Downloading the historical data from Yahoo Finance
    df = yf.download(user_input, start=start, end=end)

    # Showing the descriptive statistics of the data
    st.subheader('Data from 2000 - 2023')
    st.write(df.describe())

    # Plotting the closing price vs time chart
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Closing Price of {user_input} from 2000 to 2023')
    st.pyplot(fig)

    # Allow users to customize the moving averages
    show_ma50 = st.checkbox('Show 50-day MA')
    show_ma100 = st.checkbox('Show 100-day MA')
    show_ma200 = st.checkbox('Show 200-day MA')

    # Calculating and plotting the moving averages based on user input
    if show_ma50:
        ma50 = df.Close.rolling(50).mean()
        plt.plot(ma50, 'y', label='50-day MA')
    if show_ma100:
        ma100 = df.Close.rolling(100).mean()
        plt.plot(ma100, 'r', label='100-day MA')
    if show_ma200:
        ma200 = df.Close.rolling(200).mean()
        plt.plot(ma200, 'g', label='200-day MA')

    # Plotting the closing price
    plt.plot(df.Close, 'b', label='Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Closing Price and Moving Averages of {user_input} from 2000 to 2023')
    plt.legend()
    st.pyplot(fig)

    # Splitting the data into training and testing sets
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    # Scaling the data to the range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Creating the input and output sequences for the training data
    x_train = []
    y_train = []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Setting the model file path
    model_file_path = 'ker_model.h5'

    # Loading the pre-trained model from the file path
    model = None  # Initialize model to None

    # Check if the file exists
    if os.path.exists(model_file_path):
        model = load_model(model_file_path)

    # Testing the model on the testing data
    if model is not None:
        # Concatenating the last 100 days of the training data and the testing data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        # Scaling the final data
        input_data = scaler.fit_transform(final_df)

        # Creating the input and output sequences for the testing data
        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predicting the closing prices using the model
        y_predicted = model.predict(x_test)

        # Rescaling the predicted and actual values to the original scale
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plotting the predicted and actual prices
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'Prediction vs Original Price of {user_input} from 2000 to 2023')
        plt.legend()
        st.pyplot(fig2)
    else:
        # Displaying an error message if the model file is not found
        st.error(f"Model file '{model_file_path}' not found.")

    # User feedback section
    # User feedback section with a placeholder
    # User feedback section with directions and emoji
    st.subheader('User Feedback 📣')
    st.markdown("""
        Provide your feedback in the text area below. 
        You can share your thoughts, suggestions, or report issues. 
        Let us know how we can improve! 🚀
    """)

    # Get user feedback
    user_feedback = st.text_area('Feedback', value='Type your feedback here...')

    # Check if user entered meaningful feedback before submission
    if user_feedback != 'Type your feedback here...' and st.button('Submit Feedback'):
        # Save feedback to a file
        feedback_folder = 'user_feedback'
        os.makedirs(feedback_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        feedback_file_path = os.path.join(feedback_folder, f'feedback_{timestamp}.txt')
        
        with open(feedback_file_path, 'w') as feedback_file:
            feedback_file.write(user_feedback)

        # Process the feedback (you can store it, analyze it, etc.)
        st.success('Feedback submitted successfully! Thank you for your input! 🙌')
    elif st.button('Submit Feedback'):
        # Display a message if the user didn't provide meaningful feedback
        st.warning('Please provide meaningful feedback before submitting.')
else:
    st.warning("You need to log in to access the app.")










