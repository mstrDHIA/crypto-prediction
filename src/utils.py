import numpy as np

def scale_data(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler



# def invert_scaling(scaled_data, scaler):
#     return scaler.inverse_transform(scaled_data)

def date_to_timestamp(date_series, unit='s'):
    try:
        timestamp_series = pd.to_datetime(date_series).astype(int) // 10**9
        if unit == 'ms':
            timestamp_series = pd.to_datetime(date_series).astype(int) // 10**6
        elif unit == 'us':
            timestamp_series = pd.to_datetime(date_series).astype(int) // 10**3
        elif unit == 'ns':
            timestamp_series = pd.to_datetime(date_series).astype(int)
        return timestamp_series
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def timestamp_to_date(timestamp_series, unit='s'):
    try:
        date_series = pd.to_datetime(timestamp_series, unit=unit)
        return date_series
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, 0])  # Predicting 'Close' price
    return np.array(X), np.array(y)


def inverse_scaling(y_test, y_pred, scaler):
    y_test_expanded = np.zeros((y_test.shape[0], scaler.n_features_in_))
    y_test_expanded[:, 0] = y_test.flatten()  # Place y_test in the first column

    y_pred_expanded = np.zeros((y_pred.shape[0], scaler.n_features_in_))
    y_pred_expanded[:, 0] = y_pred.flatten()  # Place y_pred in the first column

    y_test_inv = scaler.inverse_transform(y_test_expanded)[:, 0]  # Extract the first column
    y_pred_inv = scaler.inverse_transform(y_pred_expanded)[:, 0]  # Extract the first column
    return y_test_inv, y_pred_inv