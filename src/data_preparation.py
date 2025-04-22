from src.utils import create_sequences

def data_preparation(scaled_df):
    print("Data Preparation started...")
    # Check if scaled_df is empty
    if scaled_df.empty:
        print("Error: The scaled DataFrame is empty.")
        return None, None, None, None
    #Time Step
    time_step = 10

    scaled_data = scaled_df.values
    #  Create sequences
    X, y = create_sequences(scaled_data, time_step)
    
    # Split the data into training and testing sets
    # 80% for training and 20% for testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print("Data Preparation completed.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return X_train, X_test, y_train, y_test