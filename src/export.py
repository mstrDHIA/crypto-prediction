import os
import shutil
from datetime import datetime
from src.count import get_and_increment_count
import json
def export_results(model,hyperparameters):
    counter = get_and_increment_count()

    folder_path=create_folder(counter)
    export_model(model,folder_path)
    export_model_summary(model, folder_path)

    export_hyperparameters(folder_path, hyperparameters)
    
   
    # # Copy dataset
    # if os.path.exists(dataset_path):
    #     shutil.copy(dataset_path, os.path.join(results_folder, "dataset.csv"))

    # # Copy predicted price graph
    # if os.path.exists(predicted_price_graph_path):
    #     shutil.copy(predicted_price_graph_path, os.path.join(results_folder, "predicted_price_graph.png"))

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
