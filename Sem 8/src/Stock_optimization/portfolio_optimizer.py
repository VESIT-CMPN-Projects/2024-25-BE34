import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import requests
import pymongo
from datetime import datetime
from newsapi import NewsApiClient
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import urllib.parse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import plotly.graph_objects as go


#from scratch_5 import calculate_kpis

# NewsAPI client
newsapi = NewsApiClient(api_key='Your_api_key')

# MongoDB connection details
username = 'username'
password = 'password'
username_encoded = urllib.parse.quote_plus(username)
password_encoded = urllib.parse.quote_plus(password)

# Create MongoDB URI
mongo_uri = f"mongodb+srv://{username_encoded}:{password_encoded}@cluster-kr0.co3r3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-kr0"

# Create a new client and connect to the server, disabling SSL verification
client = MongoClient(mongo_uri, server_api=ServerApi('1'), tls=True, tlsInsecure=True)

# Check connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection error: {e}")

# Select the database and collection
database = client["stock_portfolio"]
users_collection = database["user_data"]

# User authentication functions
def check_user(username, password):
    """Check if the username and password exist in MongoDB."""
    user = users_collection.find_one({'username': username, 'password': password})
    return user is not None

# User registration function
def register_user(username1, password1, name, age, occupation, location, past_portfolio, trading_preference,
                  investor_type, educational_level):
    """Register a new user and store details in MongoDB."""
    user_data = {
        'username': username1,
        'password': password1,
        'full_name': name,
        'age': age,
        'occupation': occupation,
        'location': location,
        'past_portfolio': past_portfolio,
        'trading_preference': trading_preference,
        'investor_type': investor_type,
        'educational_level': educational_level
    }
    # Insert user data into the collection
    result = users_collection.insert_one(user_data)
    print(f"Inserted user data with ID: {result.inserted_id}")

# Function to get Indian stock names from the uploaded CSV file
@st.cache_data
# Define function to get Indian stock names from the uploaded CSV file
def get_indian_stocks():
    file_path = 'All_Indian_Stocks_listed_in_nifty500.csv'
    stocks_df = pd.read_csv(file_path)
    stocks_df['Symbol'] = stocks_df['Symbol'].astype(str) + ".NS"  # Append .NS
    indian_stocks = dict(zip(stocks_df['Symbol'], stocks_df['Company Name']))
    return indian_stocks


# Function to get stock data
@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return data

def optimize_stocks():
    indian_stocks = get_indian_stocks()
    returns_data = {}

    for ticker in indian_stocks.keys():
        try:
            stock_data = yf.download(ticker, period='1y')
            if not stock_data.empty:
                daily_returns = stock_data['Close'].pct_change().dropna()
                expected_return = daily_returns.mean() * 252  # Annualized return
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
                returns_data[ticker] = {'Expected Return': expected_return, 'Volatility': volatility}
            # Removed warning for missing data
        except Exception as e:
            # You can choose to handle exceptions silently, or log them if necessary
            # For now, we skip printing anything
            pass

    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_data).T

    # Sort stocks by expected return and select top 5
    top_stocks = returns_df.sort_values(by='Expected Return', ascending=False).head(5)

    return top_stocks



def calculate_kpis(tickers):
    kpis = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        kpis[ticker] = {
            "P/E Ratio (Price to Earnings Ratio)": stock.info.get('trailingPE', None),
            "P/B Ratio(Price to Book Ratio)": stock.info.get('priceToBook', None),
            "EPS(Earning's per share)": stock.info.get('trailingEps', None),
            "Debt-to-Equity Ratio": None,
            "Dividend Yield": stock.info.get('dividendYield', None),
            "Sharpe Ratio": None
        }

        try:
            total_liabilities = stock.balance_sheet.loc['Total Liab'].values[0]
            total_equity = stock.balance_sheet.loc['Total Stockholder Equity'].values[0]
            kpis[ticker]["Debt-to-Equity Ratio"] = total_liabilities / total_equity
        except (KeyError, IndexError):
            kpis[ticker]["Debt-to-Equity Ratio"] = None

        price_data = stock.history(period='1y')['Close']
        daily_returns = price_data.pct_change().dropna()
        risk_free_rate = 0.06 / 252
        expected_return = daily_returns.mean()
        volatility = daily_returns.std()

        if volatility != 0:
            kpis[ticker]["Sharpe Ratio"] = (expected_return - risk_free_rate) / volatility
        else:
            kpis[ticker]["Sharpe Ratio"] = None

    return kpis


# Function to fetch news articles based on stock names
def fetch_stock_news(stock_names):
    articles = []
    for stock_name in stock_names:
        if not stock_name or len(stock_name.strip()) == 0:
            continue  # Skip invalid stock names
        first_word = stock_name.split()[0]
        news_data = newsapi.get_everything(q=first_word, language='en')

        if news_data['status'] == 'ok':
            for article in news_data['articles']:
                if (first_word.lower() in article['title'].lower() or
                        (article['description'] and first_word.lower() in article['description'].lower())):
                    articles.append(article)
    return articles

# Main function
def main():
    st.title('Indian Stock Portfolio Optimizer')
    st.markdown(
        "<a href='https://docs.google.com/document/d/1l8sBxi8TYTfzIKnRlsifyilBR-29pBXT3EzCGtkonFU/edit?usp=sharing' target='_blank'>Learn to Invest</a>",
        unsafe_allow_html=True,
    )
    # Initialize stock_data variable
    stock_data = {}

    # Step 1: User login page
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        username = st.text_input("Enter Username")
        password = st.text_input("Enter Password", type='password')

        if st.button("Login"):
            if check_user(username, password):
                st.session_state['logged_in'] = True
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid credentials. Please try again.")

        # User registration form
        if st.checkbox("Register New Account"):
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=18, max_value=100)
            occupation = st.text_input("Occupation")
            location = st.text_input("Location")
            past_portfolio = st.text_area("Describe Past Portfolio Experience")
            trading_preference = st.selectbox("Trading Preference",
                                              ["Day Trading", "Swing Trading", "Long-Term Investment"])
            investor_type = st.selectbox("Investor Type", ["Aggressive", "Conservative", "Moderate"])
            educational_level = st.selectbox("Educational Level", ["High School", "Undergraduate", "Postgraduate"])
            username1 = st.text_input("New Username")
            password1 = st.text_input("New Password", type="password")
            # Display the terms and conditions checkbox with a hyperlink separately
            terms_accepted = st.checkbox("I accept the terms and conditions")

            # Show the hyperlink below the checkbox
            st.markdown(
                "[View the terms and conditions](https://docs.google.com/document/d/1Ij4hi-VuPoDpxKB4CbUV3B7Q-5BTYnSbApfjGklLt28/edit?usp=sharing)")

            if st.button("Register") and terms_accepted:
                register_user(username1, password1, name, age, occupation, location, past_portfolio, trading_preference,
                              investor_type, educational_level)
                st.success("Registration successful! Please login.")

    # Step 2: Portfolio analysis (after login)
    else:
        st.sidebar.title("Portfolio Optimization Options")

        # Stock selection
        stocks = get_indian_stocks()
        selected_stocks = st.sidebar.multiselect("Select stocks", list(stocks.keys()))

        # Date selection
        today = datetime.today().strftime('%Y-%m-%d')
        start_date = st.sidebar.date_input("Select start date", value=pd.to_datetime("2020-01-01"))
        end_date = st.sidebar.date_input("Select end date", value=pd.to_datetime(today))
        if end_date <= start_date:
            raise ValueError("End date should be after the start date")
        # Analyze button
        if st.sidebar.button("Analyze"):
            if selected_stocks:
                stock_data = get_stock_data(selected_stocks, start_date, end_date)

                # Visualizing stock charts and calculating KPIs
                for ticker in selected_stocks:
                    st.subheader(f"{stocks[ticker]} ({ticker}) Stock Data")
                    if ticker in stock_data:
                        df = stock_data[ticker]
                        fig = px.line(df, x=df.index, y="Close", title=f"{stocks[ticker]} Stock Prices")
                        st.plotly_chart(fig)

                        # Calculate KPIs
                        kpis = calculate_kpis([ticker])
                        st.write("Key Performance Indicators (KPIs):")
                        for key, value in kpis[ticker].items():
                            st.write(f"{key}: {value}")

                        # Predict prices using LSTM
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

                        prediction_days = 60
                        x_train, y_train = [], []
                        for x in range(prediction_days, len(scaled_data)):
                            x_train.append(scaled_data[x - prediction_days:x, 0])
                            y_train.append(scaled_data[x, 0])

                        x_train, y_train = np.array(x_train), np.array(y_train)
                        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                        lstm_model = Sequential()
                        lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                        lstm_model.add(Dropout(0.2))
                        lstm_model.add(LSTM(units=100, return_sequences=False))
                        lstm_model.add(Dropout(0.2))
                        lstm_model.add(Dense(units=1))  # Prediction of the next closing price

                        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                        lstm_model.fit(x_train, y_train, epochs=100, batch_size=64)

                        test_data = scaled_data[-prediction_days:]
                        test_data = np.array([test_data])
                        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

                        lstm_pred = lstm_model.predict(test_data)
                        lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]
                        st.write(f"LSTM Predicted Closing Price for the Next Day: {lstm_pred}")

                        # Predict price using ARIMA
                        arima_model = ARIMA(df['Close'], order=(5, 1, 0))
                        arima_model_fit = arima_model.fit()
                        arima_forecast = arima_model_fit.forecast(steps=1)  # Ensure steps=1 to predict only the next day
                        arima_pred = arima_forecast.iloc[0]  # Use iloc[0] to get the first predicted value
                        st.write(f"ARIMA Predicted Closing Price for the Next Day: {arima_pred}")

                        # Model evaluation metrics
                        true_prices = df['Close'][-1:]  # Use the last actual price only for evaluation
                        mae_lstm = mean_absolute_error(true_prices, [lstm_pred])
                        rmse_lstm = mean_squared_error(true_prices, [lstm_pred], squared=False)
                        st.write(f"LSTM Model - MAE: {mae_lstm}, RMSE: {rmse_lstm}")

                        mae_arima = mean_absolute_error(true_prices, [arima_pred])
                        rmse_arima = mean_squared_error(true_prices, [arima_pred], squared=False)
                        st.write(f"ARIMA Model - MAE: {mae_arima}, RMSE: {rmse_arima}")

                        # Plotting Actual vs Predicted prices for both models
                        st.subheader("Actual vs Predicted Prices")
                        # Create a new DataFrame for plotting
                        last_days = df[-60:]  # Get the last 60 days of actual prices
                        fig_actual = go.Figure()

                        # Plot actual prices
                        fig_actual.add_trace(
                            go.Scatter(x=last_days.index, y=last_days['Close'], mode='lines', name='Actual Prices',
                                       line=dict(color='green')))

                        # Plot LSTM predicted price
                        fig_actual.add_trace(
                            go.Scatter(x=[df.index[-1]], y=[lstm_pred], mode='markers', name='LSTM Predicted',
                                       marker=dict(size=10, color='red')))

                        # Plot ARIMA predicted price
                        fig_actual.add_trace(
                            go.Scatter(x=[df.index[-1]], y=[arima_pred], mode='markers', name='ARIMA Predicted',
                                       marker=dict(size=10, color='blue')))

                        # Update layout
                        fig_actual.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date",
                                                 yaxis_title="Closing Price")

                        st.plotly_chart(fig_actual)

                # Relevant news articles - Displayed only once after all stocks
                st.subheader("Relevant News Articles")
                stock_names = [stocks[ticker] for ticker in selected_stocks]
                news_articles = fetch_stock_news(stock_names)

                for article in news_articles:
                    st.markdown(
                        f"""
                        <div style="border-radius: 15px; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px;">
                            <b style="font-size: 16px;">{article['title']}</b><br>
                            <p>{article['description']}</p>
                            <a href="{article['url']}" target="_blank">Read more</a>
                        </div>
                        """, unsafe_allow_html=True)

        # Optimize button
    if st.sidebar.button('Optimize'):
        top_stocks = optimize_stocks()
        if not top_stocks.empty:
            st.subheader("Top 5 Stocks with Highest Expected Return")
            st.write(top_stocks)
        else:
            st.write("No stocks available for optimization.")

        # Learn to invest link
        #st.sidebar.markdown(
         #   "[Learn to Invest](https://docs.google.com/document/d/1l8sBxi8TYTfzIKnRlsifyilBR-29pBXT3EzCGtkonFU/edit?usp=sharing)")
        # Logout button
    if st.sidebar.button("Logout"):
     st.session_state['logged_in'] = False
     st.success("You have successfully logged out.")

            # Learn to Invest link below the Logout button

            #st.experimental_rerun()



if __name__ == "__main__":
    main()
