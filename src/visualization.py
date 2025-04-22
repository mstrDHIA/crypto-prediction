import matplotlib.pyplot as plt

def plot_close_price(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], label='Close Price', linewidth=2)
    plt.title('Crypto Close Price Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_moving_averages(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], label='Close Price', linewidth=2)
    plt.plot(df['SMA_7'], label='SMA 7', linestyle='--')
    plt.plot(df['SMA_21'], label='SMA 21', linestyle='--')
    plt.plot(df['EMA_14'], label='EMA 14', linestyle=':')
    plt.title('Close Price with Moving Averages')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_daily_returns(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Return_1D'], color='orange')
    plt.title('Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()

def plot_volatility(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Volatility_7D'], color='purple')
    plt.title('7-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Std Dev)')
    plt.grid(True)
    plt.show()

def plot_rsi(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['RSI_14'], color='green')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='blue', linestyle='--', label='Oversold (30)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_test_inv, y_pred_inv):
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_inv, label='Actual Price', linewidth=2)
    plt.plot(y_pred_inv, label='Predicted Price', linestyle='--')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    return plt

def plot_correlation_matrix(df):
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
# correlation_matrix = df.corr()
# print(correlation_matrix['close'].sort_values(ascending=False))