import os
from datetime import datetime

def save_results(dataset: str, task: str, results, model, best_params = None):
    result_file_path = os.path.join("results", f"{model}_best_results_{dataset}.txt")
    with open(result_file_path, "w") as file:
        file.write(f"[{dataset} - {task}]\n")
        file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        if best_params is not None:
            file.write(f"- Best hyperparameters: {best_params}\n")
        if task == "classification":
            file.write(f"   - Accuracy: {results[0]}\n")
            file.write(f"   - Precision: {results[1]}\n")
            file.write(f"   - Recall: {results[2]}\n")
            file.write(f"   - F1: {results[3]}\n\n")
        elif task == "regression":
            file.write(f"   - MAE: {results[0]}\n")
            file.write(f"   - MSE: {results[1]}\n")
            file.write(f"   - R2: {results[2]}\n\n")

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

def save_metrics_stn(train_loss, val_loss, test_loss, task, model, dataset, val_acc = None, test_acc = None):
    if not os.path.exists("plot_data"):
        os.makedirs("plot_data")

    if train_loss:
        train_loss_file = os.path.join("plot_data", f"{model}_{dataset}_train_loss.csv")
        with open(train_loss_file, 'w') as f:
            for loss in train_loss:
                f.write(f"{loss}\n")

    if val_loss:
        val_loss_file = os.path.join("plot_data", f"{model}_{dataset}_val_loss.csv")
        with open(val_loss_file, 'w') as f:
            for loss in val_loss:
                f.write(f"{loss}\n")

    if test_loss:
        test_loss_file = os.path.join("plot_data", f"{model}_{dataset}_test_loss.csv")
        with open(test_loss_file, 'w') as f:
            for loss in test_loss:
                f.write(f"{loss}\n")

    if task == "classification":
        if val_acc:
            val_acc_file = os.path.join("plot_data", f"{model}_{dataset}_val_accuracy.csv")
            with open(val_acc_file, 'w') as f:
                for acc in val_acc:
                    f.write(f"{acc}\n")
                    
        if test_acc:
            test_acc_file = os.path.join("plot_data", f"{model}_{dataset}_test_accuracy.csv")
            with open(test_acc_file, 'w') as f:
                for mse in test_acc:
                    f.write(f"{mse}\n")