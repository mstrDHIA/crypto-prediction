from src.visualization import plot_predictions
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from src.utils import inverse_scaling
import os
class Model:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):

        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=64,return_sequences=True,))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=64,return_sequences=False,))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    def summary(self):
        return self.model.summary()
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test,scaler):
        from sklearn.metrics import mean_squared_error
        import numpy as np

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"RMSE: {rmse}")

        # # Inverse scaling for better interpretability
        y_test_inv, y_pred_inv = inverse_scaling(y_test, y_pred, scaler)
        

        # Plot actual vs predicted prices
        plot_predictions(y_test_inv, y_pred_inv)

        return rmse

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
