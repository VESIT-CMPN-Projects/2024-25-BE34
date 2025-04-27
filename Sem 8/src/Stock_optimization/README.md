Steps to run this project:

1. Install streamlit using the terminal command $ pip install streamlit
2. Install watchdog for better perfomance by using the command $ pip install watchdog
3. To finally run the project, use the command $ streamlit run portfolio_optimizer.py

   Using this project:
   1. After Logging in, firstly, you'll select the stocks that you want to analyze. This can be done by either using the drop down menu or by simply searching the stock's name
   2. Next you'll be selecting the time period that you want to consider for your stocks.
   3. Once we've clicked on "analyse", the data will be fetched and interactive visualizations will be plotted. Predicted closing prices for the next day will be displayed along with performance evaluation of both LSTM and ARIMA Models.
   4. Lastly, relevant news to the stocks you've selected will be displayed along with the links to the original news article. This feature is added to better analyse the stocks.
   5. There is also an optimise button below the analyse button that will return you the top 5 stocks along with their expected return and volatilty. This is for long term capital gains.
  
   Further updates including Mutual Funds, SIPs and breaking news related to stocks will be integrated in the next update ;)
