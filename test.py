import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
from fredapi import Fred

FRED_API_KEY = 'c8ffb85d56a65bb030644d9d02528564'
fred = Fred(api_key=FRED_API_KEY)

def fetch_stock_data(ticker, start_date="2000-11-13", end_date=None):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, interval="1d")
    # Keep the 'Close', 'High', 'Low', etc., without calculating the percentage change
    columns_to_keep = ['Close', 'High', 'Low', 'Open', 'Volume']
    stock_data = stock_data[columns_to_keep]
    
    stock_data.index = pd.to_datetime(stock_data.index)
    return stock_data.resample('Q').ffill()



def fetch_fred_data(indicators, start_date):
    fred = Fred(api_key=FRED_API_KEY)
    dfs = []

    for series_id in indicators:
        data = fred.get_series(series_id, start_date)
        df = pd.DataFrame({f"{series_id}": data})
        dfs.append(df)

    result_df = pd.concat(dfs, axis=1)

    result_df.ffill(inplace=True)

    return result_df

def calculate_velocity(data, window=10):
    data['Velocity'] = data['Close'].pct_change() * 100 
    data['Velocity'] = data['Velocity'].rolling(window=window).mean()
    return data

def calculate_magnitude(data, window=10):
    if 'High' in data.columns and 'Low' in data.columns:
        data['Magnitude'] = data['High'] - data['Low']
        data['Magnitude'] = data['Magnitude'].rolling(window=window).mean()
    else:
        print("Missing 'High' or 'Low' columns in DataFrame.")
    return data


def create_economic_data_column(data, economic_data):
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    for series_id in economic_data.columns:
        economic_data_series = economic_data[series_id]
        if economic_data_series.index.tz is not None:
            economic_data_series = economic_data_series.tz_localize(None)
        
        economic_data_series = economic_data_series.reindex(data.index, method='ffill')
        data[f"{series_id}"] = economic_data_series

    return data

def create_target_column(data, horizon=1):
    data["Future_Close"] = data["Close"].shift(-horizon * 3)
    data["Target"] = (data["Future_Close"] > data["Close"]).astype(int)
    data = data.dropna(subset=["Target"])
    return data

def generate_features(data, horizons):
    new_predictors = []

    for horizon in horizons:
        rolling_averages = data["Close"].rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]

    data = data.dropna(subset=data.columns[data.columns != "Target"])
    return data, new_predictors

def combine_data(ticker, series_ids, date):
    stock_data = fetch_stock_data(ticker, start_date=date)
    economic_data = fetch_fred_data(series_ids, date)

    # Ensure both stock and economic data have the correct timezones
    if stock_data.index.tz is not None:
        stock_data.index = stock_data.index.tz_localize(None)
    if economic_data.index.tz is not None:
        economic_data.index = economic_data.index.tz_localize(None)

    # Incorporate economic data
    stock_data = create_economic_data_column(stock_data, economic_data)

    # Prepare stock data with additional features
    stock_data = calculate_magnitude(stock_data)
    stock_data = calculate_velocity(stock_data)
    
    # Create the Target column before generating features
    stock_data = create_target_column(stock_data, horizon=1)

    # Generate features based on horizons and include economic data
    stock_data, new_predictors = generate_features(stock_data, [10, 20, 30])

    return stock_data, new_predictors + list(economic_data.columns)

def train_and_predict(ticker, series_ids, date="2000-11-13", threshold=0.5):
    data, new_predictors = combine_data(ticker, series_ids, date)
    if data.empty:
        print(f"No data available for {ticker}. Cannot proceed with prediction.")
        return None

    # Define predictors
    predictors = new_predictors + ['Magnitude', 'Velocity']

    # Prepare X, y for modeling
    X = data[predictors].dropna()
    y = (data[f"{ticker}_pct_change"] > 0).astype(int)

    # Shift y to predict next quarter's performance
    y = y.shift(-1).dropna()
    X = X.iloc[:-1, :]  # Align X with shifted y

    if len(X) < 1 or len(y) < 1:
        print("Not enough data for training and prediction.")
        return None

    # Split data, train model, and predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    latest_data = X.iloc[[-1]]

    probabilities = model.predict_proba(latest_data)
    invest_proba = probabilities[:, 1]

    if invest_proba[0] < threshold:
        decision = "Do not invest"
    else:
        decision = "Invest"

    return decision

important_series = [
    "CPIAUCNS",  # Consumer price index
    "FEDFUNDS",  # Federal funds interest rate
    "GS10",      # 10-Year treasury yield
    "M2",        # Money stock measures
    "MICH",      # UMich: inflation expectation
    "UMCSENT",   # UMich: consumer sentiment
    "UNRATE"    # Unemployment rate
]

threshold = 0.5 
prediction_vivix = train_and_predict("VIVIX", important_series, "2000-11-13", threshold)
prediction_vigax = train_and_predict("VIGAX", important_series, "2000-11-13", threshold)

print(f"VIVIX: {prediction_vivix}")
print(f"VIGAX: {prediction_vigax}")