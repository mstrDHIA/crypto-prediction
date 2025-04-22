import pandas as pd
import numpy as np
from src.visualization import plot_correlation_matrix
from src.feature_engineering import create_features
from sklearn.preprocessing import MinMaxScaler
def data_preprocessing(df):

    print("Data Preprocessing started...")
    #Data cleaning
    df=clean_data(df)

    #Data sorting if not sorted
    df=sort_data(df)

    #Data Analysis
    analyze_data(df)

    #Feature Engineering
    df=create_features(df)

    #Data Scaling
    scaled_data, scaler = scale_data(df)

    print("Data Preprocessing completed.")
    return df, scaled_data, scaler

def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(df.head())
    return df

def timestamp_to_date(timestamp_series, unit='s'):
    try:
        date_series = pd.to_datetime(timestamp_series, unit=unit)
        return date_series
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def clean_data(df):
    print("Missing values per column:")
    print(df.isnull().sum())
    df = df.dropna()
    return df

def sort_data(df):
    if 'time' in df.columns:
        if is_time_sorted(df): 
            print("Data is already sorted by time.")
        else:
            print("Data is not sorted by time. Sorting now.")
            df = df.iloc[::-1].reset_index(drop=True)
            # df = df.sort_values(by='time', ascending=True)
    return df



def is_time_sorted(df):
    if 'time' in df.columns:
        first_time = df['time'].iloc[0]
        last_time = df['time'].iloc[-1]
        return first_time <= last_time
    else:
        print("The 'time' column is not present in the DataFrame.")
        return False


def analyze_data(df):
    print("Data types:")
    print(df.dtypes)
    print("\nStatistical summary:")
    print(df.describe())
    print("\nData shape:")
    print(df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\ncorrelation Matrix:")
    plot_correlation_matrix(df)


def scale_data(df):
    print("Scaling data...")
    features = df[['close','time', 'SMA_7', 'EMA_14', 'RSI_14','Return_1D','Return_7D','Volatility_7D', 'volumeto','volumefrom','high','low','open','SMA_21']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_data, columns=['close','time', 'SMA_7', 'EMA_14', 'RSI_14','Return_1D','Return_7D','Volatility_7D', 'volumeto','volumefrom','high','low','open','SMA_21'])
    print("Scaled data:")
    print(scaled_df.head())
    print("Original data:")
    print(df.head())
    print("Scaled data shape:")
    print(scaled_df.shape)
    print("Original data shape:")
    print(df.shape)
    print("Data scaling completed.")
    return scaled_df, scaler
