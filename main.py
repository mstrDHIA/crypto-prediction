import pandas as pd
from src.data_preprocessing import load_data, clean_data, data_preprocessing
from src.feature_engineering import create_features
# from src.prediction import make_future_predictions
from src.visualization import plot_predictions, plot_moving_averages, plot_daily_returns, plot_volatility, plot_rsi,plot_close_price, plot_correlation_matrix
from src.utils import create_sequences, scale_data, date_to_timestamp, timestamp_to_date
from src.data_preparation import data_preparation
from src.model import Model
from src.export import export_results
def main():

    
    hyperparameters = {
        'epochs': 100,
        'batch_size': 32,
        'activation': 'relu',
        'optimizer': 'adam',
        'loss_function': 'mean_squared_error',
        'units': 64,
        'dropout': 0.2,
        'layers': 2,
        'validation_split': 0.2,
        'time_steps': 5  # Number of previous time steps to consider for prediction
        # 'learning_rate': 0.001,
        # 'dropout_rate': 0.2,
        # 'lstm_units': 50,
        # 'dense_units': 25
    }
    # epochs=10
    # batch_size=32


    # Load and preprocess data
    df = load_data('data/BTCUSD.csv')

    #data preprocessing
    df, scaled_df, scaler=data_preprocessing(df,800)

    X_train, X_test, y_train, y_test = data_preparation(scaled_df,hyperparameters)
    

   
    print(df.columns)
   


    # # # Initialize and train the model
    model = Model(input_shape=(X_train.shape[1], X_train.shape[2]), hyperparameters=hyperparameters)
    input_shape=(X_train.shape[1], X_train.shape[2])
    a=model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]), hyperparameters=hyperparameters)
    print("Model summary:")
    print(model.summary())
    model.train(X_train, y_train, X_test, y_test, hyperparameters)

    # # # Evaluate the model
    rmse, y_test, y_pred=model.evaluate(X_test, y_test, scaler)
    evaluation_metrics = {
        "RMSE": rmse
    }
    export_results(model, hyperparameters,y_test, y_pred,df, evaluation_metrics)
    

    




    
    # # # Inverse scaling for predictions
    # y_test_inv = invert_scaling(y_test, scaler)
    # y_pred_inv = invert_scaling(model.predict(X_test), scaler)
    # # print(f"Inverse scaled predictions: {y_pred_inv}")
    # plot_predictions(y_test_inv, y_pred_inv)

    # # # Make predictions
    # # predictions = make_future_predictions(model, df)

    # # # Visualize results
    # # plot_predictions(df, predictions)

if __name__ == "__main__":
    main()