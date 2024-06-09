# import torch
import numpy as np
from utils.data_preprocessing import load_data, preprocess_data

def run_experiment():
    print("*** Self-Tuning networks ***")
    # print(f"PyTorch version: {torch.__version__}")
    # print(f"CUDA: {torch.cuda.is_available()}")

    X_iris, y_iris = load_data("iris")
    X_iris_processed, y_iris_processed, iris_label_mapping = preprocess_data(X_iris, y_iris)

    X_student, y_student = load_data("student_dropout")
    X_student_processed, y_student_processed, student_label_mapping = preprocess_data(X_student, y_student)

    X_house, y_house = load_data("house_rent")
    X_house_processed, y_house_processed = preprocess_data(X_house, y_house, regression = True)

    print(X_house_processed)
    print(y_house_processed)

if __name__ == "__main__":
    run_experiment()