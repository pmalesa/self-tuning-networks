import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.data_preprocessing import load_data, preprocess_data, train_val_test_split
from utils.data_saver import save_metrics, save_results
from sklearn.model_selection import KFold, train_test_split
from models.nn import NeuralNetwork
import os
from datetime import datetime

def train_and_validate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, dataset: str, hidden_layers = [64, 64, 64], task = "classification"):
    # Ensure results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Initialize hyperparameters' grid for grid search
    param_grid = {
        "learning_rate": [0.001, 0.01],
        "batch_size": [16, 32],
        "epochs": [10, 50, 100],
        "betas": [(0.9, 0.999), (0.9, 0.99)],
        "eps": [1e-8, 1e-7]
    }

    best_score = float("inf") if task == "regression" else 0
    best_params = None

    # K-Fold cross-validation
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    
    for lr in param_grid["learning_rate"]:
        for batch_size in param_grid["batch_size"]:
            for epochs in param_grid["epochs"]:
                for betas in param_grid["betas"]:
                    for eps in param_grid["eps"]:
                        scores = []
                        for train_index, val_index in kf.split(X_train):
                            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                            model, _, _, _ = train_model(X_train_fold, y_train_fold, task, hidden_layers, lr, batch_size, epochs, betas, eps)
                            score = test_model(model, X_val_fold, y_val_fold, task)
                            if task == "classification":
                                score = score[0] # Accuracy
                            scores.append(score)

                        avg_score = np.mean(scores)
                        if (task == "regression" and avg_score < best_score) or (task == "classification" and avg_score > best_score):
                            best_score = avg_score
                            best_params = {
                                "learning_rate": lr,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "betas": betas,
                                "eps": eps
                            }

    # Convert test data to tensors
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_test_tensor = torch.tensor(y_test, dtype = torch.long) if task == "classification" else torch.tensor(y_test.astype(np.float32)).view(-1, 1)


    # Retrain the model with the best parameters
    model, train_loss, train_score, test_score = train_model(
        X_train,
        y_train,
        task,
        hidden_layers,
        best_params["learning_rate"],
        best_params["batch_size"],
        best_params["epochs"],
        best_params["betas"],
        best_params["eps"],
        X_test_tensor,
        y_test_tensor,
        gather_metrics = True
    )

    # Save training metrics
    save_metrics(train_loss, train_score, test_score, task, "adam", dataset)

    return model, best_params

def train_model(X_train, y_train, task, hidden_layers, lr, batch_size, epochs, betas = (0.9, 0.999), eps = 1e-08, X_test = None, y_test = None, gather_metrics = False):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    if task == "regression":
        y_train_tensor = torch.tensor(y_train.astype(np.float32)).view(-1, 1)  # Ensure correct shape for regression
    else:
        y_train_tensor = torch.tensor(y_train, dtype = torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    # Model parameters
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train)) if task == "classification" else 1

    # Initialize model, loss function and optimizer
    model = NeuralNetwork(input_size, hidden_layers, output_size, task)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, betas = betas, eps = eps)

    # Train the model
    train_loss, train_score, test_score = model.train_model(train_loader, criterion, optimizer, epochs, X_test, y_test, gather_metrics)
    return model, train_loss, train_score, test_score

def test_model(model, X_test: np.ndarray, y_test: np.ndarray, task = "classification"):
    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    if task == "regression":
        y_test_tensor = torch.tensor(y_test.astype(np.float32)).view(-1, 1)  # Ensure correct shape for regression
    else:
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for testing
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

    # Evaluate the model
    return model.evaluate(test_loader)

def run_adam_experiment():
    print("*------* Neural Network with Adam Optimizer *------*")

    # Load the datasets
    X_iris, y_iris = load_data("iris")
    X_iris_processed, y_iris_processed, iris_label_mapping = preprocess_data(X_iris, y_iris)

    X_student, y_student = load_data("student_dropout")
    X_student_processed, y_student_processed, student_label_mapping = preprocess_data(X_student, y_student)

    X_house, y_house = load_data("house_rent")
    X_house_processed, y_house_processed = preprocess_data(X_house, y_house, regression = True)

    # Split the datasets into training, validation and test subsets
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris_processed, y_iris_processed, test_size = 0.1, random_state = 42)
    X_student_train, X_student_test, y_student_train, y_student_test = train_test_split(X_student_processed, y_student_processed, test_size = 0.1, random_state = 42)
    X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(X_house_processed, y_house_processed, test_size = 0.1, random_state = 42)

    # ---------------------------------------------------------------------

    # Iris dataset experiment
    iris_hidden_layers = [16, 16, 16] # Can be [16, 8] or [8, 4] as well
    model_iris, best_params = train_and_validate(X_iris_train, y_iris_train, X_iris_test, y_iris_test, "iris", iris_hidden_layers, "classification")
    results = test_model(model_iris, X_iris_test, y_iris_test, "classification")
    save_results("IRIS", "classification", best_params, results, "adam")

    # Student dropout dataset experiment
    student_hidden_layers = [64, 32, 16]
    model_student, best_params = train_and_validate(X_student_train, y_student_train, X_student_test, y_student_test, "student", student_hidden_layers, "classification")
    results = test_model(model_student, X_student_test, y_student_test, "classification")
    save_results("STUDENT_DROPOUT", "classification", best_params, results, "adam")

    # House rent dataset experiment
    house_hidden_layers = [2048, 1024, 512, 256, 128]
    model_house, best_params = train_and_validate(X_house_train, y_house_train, X_house_test, y_house_test, "house", house_hidden_layers, "regression")
    results = test_model(model_house, X_house_test, y_house_test, "regression")
    save_results("HOUSE_RENT", "regression", best_params, results, "adam")

if __name__ == "__main__":
    run_adam_experiment()
