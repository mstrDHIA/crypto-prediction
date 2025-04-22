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
        'epochs': 10,
        'batch_size': 32,
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
    df, scaled_df, scaler=data_preprocessing(df)

    X_train, X_test, y_train, y_test = data_preparation(scaled_df)
    

   
    print(df.columns)
   


    # # # Initialize and train the model
    model = Model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print("Model summary:")
    print(model.summary())
    model.train(X_train, y_train, X_test, y_test, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'])

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