import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.data_preprocessing import load_data, preprocess_data, train_val_test_split
from utils.data_saver import save_metrics, save_results
from sklearn.model_selection import train_test_split
from models.stn import train_stn_model
import os
from datetime import datetime
import argparse

def get_data_loaders(data, batch_size = 16, shuffle = True, task = "classification"):
    X_train_tensor = torch.tensor(data[0].astype(np.float32))
    X_val_tensor = torch.tensor(data[1].astype(np.float32))
    X_test_tensor = torch.tensor(data[2].astype(np.float32))

    if task == "classification":
        y_train_tensor = torch.tensor(data[3], dtype = torch.long)
        y_val_tensor = torch.tensor(data[4], dtype = torch.long)
        y_test_tensor = torch.tensor(data[5], dtype = torch.long)
    else:
        y_train_tensor = torch.tensor(data[3].astype(np.float32))
        y_val_tensor = torch.tensor(data[4].astype(np.float32))
        y_test_tensor = torch.tensor(data[5].astype(np.float32))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)

    return train_loader, val_loader, test_loader

def run_stn_experiment(delta_stn = False):
    if delta_stn:
        print("*------* Delta Self-Tuning Neural Network *------*")
    else:
        print("*------* Self-Tuning Neural Network *------*")

    # Load the datasets
    X_iris, y_iris = load_data("iris")
    X_iris_processed, y_iris_processed, iris_label_mapping = preprocess_data(X_iris, y_iris)

    X_student, y_student = load_data("student_dropout")
    X_student_processed, y_student_processed, student_label_mapping = preprocess_data(X_student, y_student)

    X_house, y_house = load_data("house_rent")
    X_house_small = X_house.sample(frac = 0.25)
    y_house_small = y_house.iloc[X_house_small.index]
    X_house_processed, y_house_processed = preprocess_data(X_house_small, y_house_small, regression = True)

    # Split the datasets into training, validation and test subsets
    X_iris_train, X_iris_val, X_iris_test, y_iris_train, y_iris_val, y_iris_test = train_val_test_split(X_iris_processed, y_iris_processed, validate = True, random_state = 42)
    X_student_train, X_student_val, X_student_test, y_student_train, y_student_val, y_student_test = train_val_test_split(X_student_processed, y_student_processed, validate = True, random_state = 42)
    X_house_train, X_house_val, X_house_test, y_house_train, y_house_val, y_house_test = train_val_test_split(X_house_processed, y_house_processed, validate = True, random_state = 42)

    iris_data = (X_iris_train, X_iris_val, X_iris_test, y_iris_train, y_iris_val, y_iris_test)
    student_data = (X_student_train, X_student_val, X_student_test, y_student_train, y_student_val, y_student_test)
    house_data = (X_house_train, X_house_val, X_house_test, y_house_train, y_house_val, y_house_test)

    iris_input_size = X_iris_train.shape[1]
    iris_output_size = len(np.unique(y_iris_train))
    student_input_size = X_student_train.shape[1]
    student_output_size = len(np.unique(y_student_train))
    house_input_size = X_house_train.shape[1]
    house_output_size = 1

    # ---------------------------------------------------------------------
    
    # Iris dataset experiment
    iris_train_loader, iris_val_loader, iris_test_loader = get_data_loaders(iris_data)
    train_stn_model(iris_input_size, iris_output_size, iris_train_loader, iris_val_loader, iris_test_loader, hidden_sizes = [16, 16, 16], dataset = "iris", delta_stn = delta_stn)

    # Student dropout dataset experiment
    student_train_loader, student_val_loader, student_test_loader = get_data_loaders(student_data)
    train_stn_model(student_input_size, student_output_size, student_train_loader, student_val_loader, student_test_loader, hidden_sizes = [64, 32, 16], dataset = "student", delta_stn = delta_stn)

    # House rent dataset experiment
    house_train_loader, house_val_loader, house_test_loader = get_data_loaders(house_data, task = "regression")
    train_stn_model(house_input_size, house_output_size, house_train_loader, house_val_loader, house_test_loader, hidden_sizes = [2048, 1024, 512], dataset = "house", task = "regression", delta_stn = delta_stn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run STN experiment")
    parser.add_argument("--delta_stn", action = "store_true", help = "Set this flag to use delta STN")
    args = parser.parse_args()
    run_stn_experiment(delta_stn = args.delta_stn)
