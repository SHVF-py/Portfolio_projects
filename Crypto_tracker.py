import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import gradio as gr

# Function to fetch historical data
def get_historical_data(crypto_id, vs_currency, days):
    base_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to create lagged features
def create_lagged_features(prices, lag=3):
    df = pd.DataFrame({"price": prices})
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df["price"].shift(i)
    df.dropna(inplace=True)
    return df

# Function to fetch, train, and predict
def predict_prices(crypto_id, days_to_predict):
    vs_currency = "usd"
    days = 60  # Fetch last 60 days of data for training

    # Fetch data
    data = get_historical_data(crypto_id, vs_currency, days)
    if not data:
        return f"Error fetching data for {crypto_id}. Please check the ID and try again.", None, None

    prices = [entry[1] for entry in data["prices"]]  # Extract prices

    # Create lagged features
    lagged_data = create_lagged_features(prices, lag=3)
    X = lagged_data.drop("price", axis=1).values
    y = lagged_data["price"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate R-squared value
    y_pred = model.predict(X_test)
    r_squared = r2_score(y_test, y_pred)

    # Predict next 'days_to_predict' days
    recent_data = X[-1].reshape(1, -1)
    forecast = []
    for _ in range(days_to_predict):
        next_price = model.predict(recent_data)[0]
        forecast.append(next_price)
        recent_data = np.append(recent_data[:, 1:], next_price).reshape(1, -1)

    # Plot historical prices and predictions
    plt.figure(figsize=(10, 6))
    zoom_start = max(0, len(prices) - 200)  # Show only the last 200 data points for historical prices
    plt.plot(range(zoom_start, len(prices)), prices[zoom_start:], label="Historical Prices")
    plt.plot(range(len(prices), len(prices) + days_to_predict), forecast, label="Forecasted Prices", color="red")
    plt.legend()
    plt.title(f"{crypto_id.capitalize()} Price Forecast")
    plt.xlabel("Time Steps")
    plt.ylabel(f"Price in {vs_currency.upper()}")
    plt.tight_layout()
    plt.savefig("crypto_forecast.png")
    plt.close()

    # Return results
    forecast_str = "\n".join([f"Day {i+1}: ${price:.2f}" for i, price in enumerate(forecast)])
    return f"Forecasted Prices for {days_to_predict} days:\n{forecast_str}", "crypto_forecast.png", f"{r_squared * 100:.2f}%"

# Gradio Interface
def gradio_interface(crypto_id, days_to_predict):
    message, graph_path, r_squared = predict_prices(crypto_id, days_to_predict)
    if graph_path:
        return message, graph_path, r_squared
    else:
        return message, None, None

# Launch Gradio
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Cryptocurrency ID (e.g., bitcoin, ethereum)"),
        gr.Slider(minimum=1, maximum=14, step=1, value=3, label="Number of Days to Predict")  # Slider for number of days
    ],
    outputs=[
        gr.Textbox(label="Prediction Results"),
        gr.Image(label="Forecast Graph"),
        gr.Textbox(label="Estimated Prediction Accuracy (R-squared)")
    ],
    title="Cryptocurrency Price Predictor",
    description="Enter the ID of a cryptocurrency and select the number of days you want to predict (1 to 14)."
)

if __name__ == "__main__":
    interface.launch(share=True)
