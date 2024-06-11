import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(model_name):
    plot_data_dir = "plot_data"
    files = {
        "iris_train_loss": os.path.join(plot_data_dir, f"{model_name}_iris_train_loss.csv"),
        "iris_train_accuracy": os.path.join(plot_data_dir, f"{model_name}_iris_train_accuracy.csv"),
        "iris_test_accuracy": os.path.join(plot_data_dir, f"{model_name}_iris_test_accuracy.csv"),
        "student_train_loss": os.path.join(plot_data_dir, f"{model_name}_student_train_loss.csv"),
        "student_train_accuracy": os.path.join(plot_data_dir, f"{model_name}_student_train_accuracy.csv"),
        "student_test_accuracy": os.path.join(plot_data_dir, f"{model_name}_student_test_accuracy.csv"),
        "house_train_loss": os.path.join(plot_data_dir, f"{model_name}_house_train_loss.csv"),
        "house_test_mse": os.path.join(plot_data_dir, f"{model_name}_house_test_mse.csv")
    }

    if not any(os.path.exists(file) for file in files.values()):
        print(f"No data available for model '{model_name}'. Ensure that training was completed and all metrics were saved.")
        return
    

    # Iris plots
    plt.figure("Iris Dataset", figsize = (12, 8))
    if os.path.exists(files["iris_train_loss"]):
        iris_train_loss = pd.read_csv(files["iris_train_loss"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(iris_train_loss, color = "blue")
        plt.title("Iris Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["iris_train_accuracy"]):
        iris_train_accuracy = pd.read_csv(files["iris_train_accuracy"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(iris_train_accuracy, color = "blue")
        plt.title("Iris Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    if os.path.exists(files["iris_test_accuracy"]):
        iris_test_accuracy = pd.read_csv(files["iris_test_accuracy"], header = None)
        plt.subplot(2, 2, 3)
        plt.plot(iris_test_accuracy, color = "blue")
        plt.title("Iris Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    plt.tight_layout()

    # Student plots
    plt.figure("Student Dropout Dataset", figsize = (12, 8))
    if os.path.exists(files["student_train_loss"]):
        student_train_loss = pd.read_csv(files["student_train_loss"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(student_train_loss, color = "red")
        plt.title("Student Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["student_train_accuracy"]):
        student_train_accuracy = pd.read_csv(files["student_train_accuracy"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(student_train_accuracy, color = "red")
        plt.title("Student Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    if os.path.exists(files["student_test_accuracy"]):
        student_test_accuracy = pd.read_csv(files["student_test_accuracy"], header = None)
        plt.subplot(2, 2, 3)
        plt.plot(student_test_accuracy, color = "red")
        plt.title("Student Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    plt.tight_layout()

    # House plots
    plt.figure("House Rent Dataset", figsize = (12, 8))
    if os.path.exists(files["house_train_loss"]):
        house_train_loss = pd.read_csv(files["house_train_loss"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(house_train_loss, color = "magenta")
        plt.title("House Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["house_test_mse"]):
        house_test_mse = pd.read_csv(files["house_test_mse"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(house_test_mse, color = "magenta")
        plt.title("House Training Mean Square Error")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def plot_metrics_stn(model_name):
    plot_data_dir = "plot_data"
    files = {
        "iris_train_loss": os.path.join(plot_data_dir, f"{model_name}_iris_train_loss.csv"),
        "iris_val_loss": os.path.join(plot_data_dir, f"{model_name}_iris_val_loss.csv"),
        "iris_test_loss": os.path.join(plot_data_dir, f"{model_name}_iris_test_loss.csv"),
        "iris_val_accuracy": os.path.join(plot_data_dir, f"{model_name}_iris_val_accuracy.csv"),
        "iris_test_accuracy": os.path.join(plot_data_dir, f"{model_name}_iris_test_accuracy.csv"),
        "student_train_loss": os.path.join(plot_data_dir, f"{model_name}_student_train_loss.csv"),
        "student_val_loss": os.path.join(plot_data_dir, f"{model_name}_student_val_loss.csv"),
        "student_test_loss": os.path.join(plot_data_dir, f"{model_name}_student_test_loss.csv"),
        "student_val_accuracy": os.path.join(plot_data_dir, f"{model_name}_student_val_accuracy.csv"),
        "student_test_accuracy": os.path.join(plot_data_dir, f"{model_name}_student_test_accuracy.csv"),
        "house_train_loss": os.path.join(plot_data_dir, f"{model_name}_house_train_loss.csv"),
        "house_val_loss": os.path.join(plot_data_dir, f"{model_name}_house_val_loss.csv"),
        "house_test_loss": os.path.join(plot_data_dir, f"{model_name}_house_test_loss.csv")
    }

    if not any(os.path.exists(file) for file in files.values()):
        print(f"No data available for model '{model_name}'. Ensure that training was completed and all metrics were saved.")
        return

    # Iris plots
    plt.figure("Iris Dataset Loss", figsize = (16, 8))
    if os.path.exists(files["iris_train_loss"]):
        iris_train_loss = pd.read_csv(files["iris_train_loss"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(iris_train_loss, color = "blue")
        plt.title("Iris Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["iris_val_loss"]):
        iris_val_loss = pd.read_csv(files["iris_val_loss"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(iris_val_loss, color = "blue")
        plt.title("Iris Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["iris_test_loss"]):
        iris_test_loss = pd.read_csv(files["iris_test_loss"], header = None)
        plt.subplot(2, 2, 3)
        plt.plot(iris_test_loss, color = "blue")
        plt.title("Iris Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    plt.tight_layout()

    plt.figure("Iris Dataset Accuracy", figsize = (16, 8))
    if os.path.exists(files["iris_val_accuracy"]):
        iris_val_accuracy = pd.read_csv(files["iris_val_accuracy"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(iris_val_accuracy, color = "blue")
        plt.title("Iris Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    if os.path.exists(files["iris_test_accuracy"]):
        iris_test_accuracy = pd.read_csv(files["iris_test_accuracy"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(iris_test_accuracy, color = "blue")
        plt.title("Iris Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    plt.tight_layout()

    # Student plots
    plt.figure("Student Dropout Dataset Loss", figsize = (16, 8))
    if os.path.exists(files["student_train_loss"]):
        student_train_loss = pd.read_csv(files["student_train_loss"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(student_train_loss, color = "red")
        plt.title("Student Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["student_val_loss"]):
        student_val_loss = pd.read_csv(files["student_val_loss"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(student_val_loss, color = "red")
        plt.title("Student Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["student_test_loss"]):
        student_test_loss = pd.read_csv(files["student_test_loss"], header = None)
        plt.subplot(2, 2, 3)
        plt.plot(student_test_loss, color = "red")
        plt.title("Student Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    plt.tight_layout()

    plt.figure("Student Dropout Dataset Accuracy", figsize = (16, 8))
    if os.path.exists(files["student_val_accuracy"]):
        student_val_accuracy = pd.read_csv(files["student_val_accuracy"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(student_val_accuracy, color = "red")
        plt.title("Student Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    if os.path.exists(files["student_test_accuracy"]):
        student_test_accuracy = pd.read_csv(files["student_test_accuracy"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(student_test_accuracy, color = "red")
        plt.title("Student Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    plt.tight_layout()

    # House plots
    plt.figure("House Rent Dataset", figsize = (12, 8))
    if os.path.exists(files["house_train_loss"]):
        house_train_loss = pd.read_csv(files["house_train_loss"], header = None)
        plt.subplot(2, 2, 1)
        plt.plot(house_train_loss, color = "magenta")
        plt.title("House Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["house_val_loss"]):
        house_val_loss = pd.read_csv(files["house_val_loss"], header = None)
        plt.subplot(2, 2, 2)
        plt.plot(house_val_loss, color = "magenta")
        plt.title("House Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if os.path.exists(files["house_test_loss"]):
        house_test_loss = pd.read_csv(files["house_test_loss"], header = None)
        plt.subplot(2, 2, 3)
        plt.plot(house_test_loss, color = "magenta")
        plt.title("House Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/make_plots.py <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    if model_name == "adam":
        plot_metrics(model_name)
    else:
        plot_metrics_stn(model_name)