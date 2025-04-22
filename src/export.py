import os
import shutil
from datetime import datetime
from src.count import get_and_increment_count
import json
import matplotlib.pyplot as plt

def export_results(model,hyperparameters,y_test, y_pred,df,evaluation_metrics):
    counter = get_and_increment_count()

    folder_path=create_folder(counter)
    export_model(model,folder_path)
    export_model_summary(model, folder_path)

    export_hyperparameters(folder_path, hyperparameters)


    plot_and_save_predictions(y_test, y_pred, folder_path)
    # Export the DataFrame
    export_dataframe(folder_path, df)
    export_evaluation_metrics(folder_path, evaluation_metrics)
   
    # # Save evaluation metrics
    # evaluation_metrics_path = os.path.join(results_folder, "evaluation_metrics.txt")
    # with open(evaluation_metrics_path, "w") as f:
    #     for key, value in evaluation_metrics.items():
    #         f.write(f"{key}: {value}\n")

    # print(f"Results exported to {results_folder}")


def create_folder(counter):
    # Check if the 'results' folder exists, otherwise create it
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Create a subfolder with the name "datetime+count"
    folder_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{counter}"
    subfolder_path = os.path.join(results_folder, folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    print(f"Folder created: {subfolder_path}")
    return subfolder_path


def export_model(model, folder_path):
    # Save the model architecture and weights
    model_path = os.path.join(folder_path, "model.h5")
    model.save_model(model_path)
    print(f"Model exported to {model_path}")

def export_model_summary(model, folder_path):
    # Save the model summary
    summary_path = os.path.join(folder_path, "model_summary.txt")
    
    model.save_summary(summary_path)
    print(f"Model summary exported to {summary_path}")


def export_hyperparameters(folder_path, hyperparameters):
    # Save hyperparameters as a JSON file
    hyperparameters_path = os.path.join(folder_path, "hyperparameters.json")
    with open(hyperparameters_path, "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters exported to {hyperparameters_path}")


def plot_and_save_predictions(y_test, y_pred, folder_path):
    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 6))
    plt.plot(y_test, label='Actual Price', linewidth=2)
    plt.plot(y_pred, label='Predicted Price', linestyle='--')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot to the results folder
    graph_path = os.path.join(folder_path, "predicted_price_graph.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Predicted price graph saved to {graph_path}")


def export_dataframe(folder_path, df):
    # Save the DataFrame as a CSV file in the results folder
    dataframe_path = os.path.join(folder_path, "dataframe.csv")
    df.to_csv(dataframe_path, index=False)
    print(f"DataFrame exported to {dataframe_path}")


def export_evaluation_metrics(folder_path, evaluation_metrics):
    # Save evaluation metrics as a JSON file
    metrics_path = os.path.join(folder_path, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_metrics, f, indent=4)
    print(f"Evaluation metrics exported to {metrics_path}")