from src.visualization import plot_predictions
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from src.utils import inverse_scaling
import os
class Model:
    def __init__(self, input_shape, hyperparameters):
        self.model = self.build_model(input_shape, hyperparameters)

    def build_model(self, input_shape, hyperparameters):

        model = Sequential()
        model.add(Input(shape=input_shape))
        # for _ in range(hyperparameters['layers']-1):
        #     if _ == 0:
        #         model.add(LSTM(units=hyperparameters['units'], return_sequences=True))
        #         model.add(Dropout(hyperparameters['dropout']))
        #     else:
                
        #         model.add(LSTM(units=hyperparameters['units'],return_sequences=False,))
        #         model.add(Dropout(hyperparameters['dropout']))
        
        # model.add(LSTM(units=64,return_sequences=False,))
        
        # model.add(Dropout(0.2))
        # New Dense layer
        model.add(LSTM(units=hyperparameters['units'], return_sequences=False))
        model.add(Dropout(hyperparameters['dropout']))
        # model.add(LSTM(units=hyperparameters['units'],return_sequences=False,))
        # model.add(Dropout(hyperparameters['dropout']))
        model.add(Dense(units=32, activation=hyperparameters['activation']))  # Added Dense layer with 32 units
        
        model.add(Dense(1))  # Output layer

        model.compile(optimizer=hyperparameters['optimizer'], loss=hyperparameters['loss_function'])
        return model
    def summary(self):
        return self.model.summary()
    def train(self, X_train, y_train, X_val, y_val,hyperparameters):
        early_stop = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=10,         # Stop training after 10 epochs with no improvement
            restore_best_weights=True,  # Restore the best weights after stopping
            verbose=1            # Print messages when stopping
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=hyperparameters['epochs'],
            batch_size=hyperparameters['batch_size'],
            validation_data=(X_val, y_val),
            # callbacks=[early_stop],  # Add the EarlyStopping callback
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test,scaler):
        from sklearn.metrics import mean_squared_error
        import numpy as np

        y_pred = self.model.predict(X_test)
        y_pred = y_pred.flatten()  # Flatten the predictions to match the shape of y_test
        print(f"Predicted values: {y_pred}")
        print(f"Actual values: {y_test}")
        print(f"Predicted values shape: {y_pred.shape}")
        print(f"Actual values shape: {y_test.shape}")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"RMSE: {rmse}")

        # # Inverse scaling for better interpretability
        y_test_inv, y_pred_inv = inverse_scaling(y_test, y_pred, scaler)
        

        # Plot actual vs predicted prices
        plot_predictions(y_test_inv, y_pred_inv)
        # print

        return rmse, y_test_inv, y_pred_inv

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, filepath):
        # model_path = os.path.join(folder_path, "model.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def save_summary(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))
        print(f"Model summary saved to {filepath}")
