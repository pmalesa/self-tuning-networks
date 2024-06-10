import os
from datetime import datetime

def save_results(dataset: str, task: str, best_params, results, model):
    result_file_path = os.path.join("results", f"{model}_best_results_{dataset}.txt")
    with open(result_file_path, "w") as file:
        file.write(f"[{dataset} - {task}]\n")
        file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        file.write(f"- Best hyperparameters: {best_params}\n")
        if task == "classification":
            file.write(f"   - Accuracy: {results[0]}\n")
            file.write(f"   - Precision: {results[1]}\n")
            file.write(f"   - Recall: {results[2]}\n")
            file.write(f"   - F1: {results[3]}\n\n")
        elif task == "regression":
            file.write(f"   - MSE: {results}\n\n")

def save_metrics(train_loss, train_score, test_score, task, model, dataset):
    if not os.path.exists("plot_data"):
        os.makedirs("plot_data")

    train_loss_file = os.path.join("plot_data", f"{model}_{dataset}_train_loss.csv")
    with open(train_loss_file, 'w') as f:
        for loss in train_loss:
            f.write(f"{loss}\n")

    if train_score:
        train_score_file = os.path.join("plot_data", f"{model}_{dataset}_train_accuracy.csv")
        with open(train_score_file, 'w') as f:
            for acc in train_score:
                f.write(f"{acc}\n")

    if test_score:
        if task == "classification":
            test_accuracy_file = os.path.join("plot_data", f"{model}_{dataset}_test_accuracy.csv")
            with open(test_accuracy_file, 'w') as f:
                for acc in test_score:
                    f.write(f"{acc}\n")
        elif task == "regression":
            test_mse_file = os.path.join("plot_data", f"{model}_{dataset}_test_mse.csv")
            with open(test_mse_file, 'w') as f:
                for mse in test_score:
                    f.write(f"{mse}\n")