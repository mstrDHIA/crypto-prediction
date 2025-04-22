from src.visualization import plot_predictions, plot_moving_averages, plot_daily_returns, plot_volatility, plot_rsi,plot_close_price



def create_features(df):
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_21'] = df['close'].rolling(window=21).mean()
    df['EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['Return_1D'] = df['close'].pct_change()
    df['Return_7D'] = df['close'].pct_change(periods=7)
    df['Volatility_7D'] = df['close'].rolling(window=7).std()
    df['RSI_14'] = compute_rsi(df['close'], 14)
    print("Feature engineering complete.")
    df.dropna(inplace=True)  # Drop rows with NaN values after feature creation
    # visualize_features(df)
    print(df.head())
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def visualize_features(df):
    plot_moving_averages(df)
    plot_daily_returns(df)
    plot_volatility(df)
    plot_rsi(df)
