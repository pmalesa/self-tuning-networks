import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, task = "classification"):
        super(NeuralNetwork, self).__init__()
        self.task = task

        # Initialize layers
        self.layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)
    
    def train_model(self, train_loader, criterion, optimizer, epochs, X_test = None, y_test = None, gather_metrics = False):
        self.train()
        train_loss = []
        train_score = []
        test_score = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if self.task == "classification":
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            train_loss.append(epoch_loss)

            if self.task == "classification":
                epoch_accuracy = correct / total
                train_score.append(epoch_accuracy)
            else:
                epoch_accuracy = None

            if gather_metrics:
                with torch.no_grad():
                    if X_test is not None and y_test is not None:
                        self.eval()
                        test_outputs = self(X_test)
                        if self.task == "classification":
                            _, test_preds = torch.max(test_outputs, 1)
                            test_acc = accuracy_score(y_test, test_preds)
                            test_score.append(test_acc)
                        elif self.task == "regression":
                            test_mse = mean_squared_error(y_test, test_outputs)
                            test_score.append(test_mse)

            print("=" * 80)
            if self.task == "classification":
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Train accuracy: {epoch_accuracy:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
            print("=" * 80)

        return train_loss, train_score, test_score

    def evaluate(self, test_loader):
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                if self.task == "classification":
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.numpy())
                    all_labels.extend(labels.numpy())
                elif self.task == "regression":
                    preds = outputs.squeeze().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
        
        if self.task == "classification":
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average = "weighted", zero_division = np.nan)
            recall = recall_score(all_labels, all_preds, average = "weighted", zero_division = np.nan)
            f1 = f1_score(all_labels, all_preds, average = "weighted", zero_division = np.nan)
            return (accuracy, precision, recall, f1)

        elif self.task == "regression":
            mse = mean_squared_error(all_labels, all_preds)
            return mse