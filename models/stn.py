from lib.stn.layers.linear import *
from lib.stn.utils.dropout_utils import dropout
from lib.stn.base_model import StnModel

from lib.stn.hyper.container import HyperContainer
from lib.stn.base_step_optimizer import *
from lib.stn.base_trainer import *

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import random
import torch
import wandb
import yaml
import sys
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from utils.data_saver import save_metrics_stn, save_results

# Initialize CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = False
cudnn.deterministic = True

class StnNeuralNetwork(StnModel):
    def __init__(self, input_size, hidden_sizes, output_size, num_hyper, h_container, use_bias = True):
        if len(hidden_sizes) != 3:
            print("[ERROR] StnNeuralNetwork class supports only three-layer networks.")
        super(StnNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layer_structure = [input_size, *hidden_sizes, output_size]
        self.num_hyper = num_hyper
        self.h_container = h_container
        self.use_bias = use_bias

        self.layers = nn.ModuleList(
            [StnLinear(self.layer_structure[i], self.layer_structure[i + 1], num_hyper = num_hyper, bias = use_bias)
             for i in range(len(self.layer_structure) - 1)]
        )

    def get_layers(self):
        return self.layers
    
    def forward(self, x, h_net, h_tensor):
        x = x.view(-1, self.input_size)

        if "dropout0" in self.h_container.h_dict:
            # at one point x_transformed becomes tensor of 'nan's
            x_transformed = self.h_container.transform_perturbed_hyper(h_tensor, "dropout0")
            """if torch.isnan(x_transformed).any():
                print(f'[inside forward()]: {h_tensor}')"""
            x = dropout(x, x_transformed, self.training)

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, h_net)
            x = F.relu(x)
            if f"dropout{i+1}" in self.h_container.h_dict:
                x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, f"dropout{i+1}"), self.training)

        x = self.layers[-1](x, h_net)

        if self.output_size == 1:
            x = x.squeeze()

        return x
    
def train_stn_model(input_size, output_size, train_loader, val_loader, test_loader, dataset, 
                    hidden_sizes = [64, 64, 64], task = "classification", delta_stn = False):
    # Load parameters from config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        parameters = config["stn"]

    # Initialize metrics
    train_loss = []
    val_loss = []
    test_loss = []
    val_acc = [] if task == "classification" else None
    test_acc = [] if task == "classification" else None
    all_preds = []
    all_labels = []
    metrics = {
        "accuracy": -1,
        "precision": -1,
        "recall": -1,
        "f1": -1
    }
    metrics_reg = {
        "mae": -1,
        "mse": -1,
        "r2": -1
    }

    # Configure hyperparameters
    h_container = HyperContainer(device)
    if parameters["tune_input_dropout"]:
        h_container.register(
            name = "dropout0",
            value = parameters["initial_dropout_value"],
            scale = parameters["initial_dropout_scale"],
            min_range = 0., max_range = 0.95,
            discrete = False, same_perturb_mb = False
        )

    if parameters["tune_dropout"]:
        h_container.register(
            "dropout1",
            parameters["initial_dropout_value"],
            parameters["initial_dropout_scale"],
            min_range = 0., max_range = 0.95,
            discrete = False, same_perturb_mb = False
        )
        h_container.register(
            "dropout2",
            parameters["initial_dropout_value"],
            parameters["initial_dropout_scale"],
            min_range = 0., max_range = 0.95,
            discrete = False, same_perturb_mb = False
        )

    num_hyper = h_container.get_size()

    # Define models and optimizers
    torch.manual_seed(parameters["model_seed"])
    np.random.seed(parameters["model_seed"])
    random.seed(parameters["model_seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed(parameters["model_seed"])
        torch.cuda.manual_seed_all(parameters["model_seed"])

    model = StnNeuralNetwork(input_size = input_size, hidden_sizes = hidden_sizes, output_size = output_size,
                             h_container = h_container, num_hyper = num_hyper)

    model = model.to(device)
    if task == "classification":
        criterion = nn.CrossEntropyLoss(reduction = "mean").to(device)
    elif task == "regression":
        criterion = nn.MSELoss(reduction = "mean").to(device)
    else:
        raise ValueError("Task must be either 'classification' or 'regression'")

    if delta_stn:
        model_general_optimizer = torch.optim.SGD(model.get_general_parameters(),
                                                lr = parameters["train_lr"],
                                                momentum = 0.9)
        model_response_optimizer = torch.optim.SGD(model.get_response_parameters(),
                                                lr = parameters["train_lr"],
                                                momentum = 0.9)
        hyper_optimizer = torch.optim.RMSprop([h_container.h_tensor], lr = parameters["valid_lr"])
        scale_optimizer = torch.optim.RMSprop([h_container.h_scale], lr = parameters["scale_lr"])

        stn_step_optimizer = DeltaStnStepOptimizer(model, model_general_optimizer, model_response_optimizer,
                                                hyper_optimizer, scale_optimizer, criterion, h_container,
                                                parameters["tune_scales"], parameters["entropy_weight"], parameters["linearize"])
    else:
        model_optimizer = torch.optim.SGD(model.parameters(), lr = parameters["train_lr"], momentum = 0.9)
        hyper_optimizer = torch.optim.RMSprop([h_container.h_tensor], lr = parameters["valid_lr"])
        scale_optimizer = torch.optim.RMSprop([h_container.h_scale], lr = parameters["scale_lr"])
        stn_step_optimizer = StnStepOptimizer(model, model_optimizer, hyper_optimizer, scale_optimizer, criterion,
                                                h_container, parameters["tune_scales"], parameters["entropy_weight"])
        
    # -------------------------------------- Evaluation functions --------------------------------------
    def delta_stn_per_epoch_evaluate(current_epoch, train_score = None):
        def evaluate(loader, test = False):
            model.eval()
            correct = total = loss = 0.
            with torch.no_grad():
                for data in loader:
                    features, labels = data[0].to(device), data[1].to(device)
                    repeated_h_tensor = model.h_container.h_tensor.unsqueeze(0).repeat((features.shape[0], 1))
                    pred = model(features, repeated_h_tensor - repeated_h_tensor.detach(), repeated_h_tensor)
                    if task == "classification":
                        loss += F.cross_entropy(pred.float(), labels.long(), reduction = "sum").item()
                        hard_pred = torch.max(pred, 1)[1]
                        correct += (hard_pred == labels).sum().item()
                        if not test:
                            all_preds.extend(hard_pred.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    elif task == "regression":
                        loss += F.mse_loss(pred.float(), labels.float(), reduction = "sum").item()
                        if not test:
                            all_preds.extend(pred.float().cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    total += labels.size(0)
            
            if task == "classification":
                accuracy = correct / float(total)
                return loss / float(total), accuracy
            return loss / float(total)

        # if train_score is None:
        train_score = evaluate(train_loader)
        val_score = evaluate(val_loader)
        test_score = evaluate(test_loader, test = True)

        print("=" * 110)
        if task == "classification":
            print("Train Epoch: {} | Train Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}"
                " | Test Loss: {:.3f} | Test Acc: {:.3f}".format(current_epoch, train_score[0], val_score[0],
                                                                val_score[1], test_score[0], test_score[1]))
        else:
            print("Train Epoch: {} | Train Loss: {:.3f} | Val Loss: {:.3f} | Test Loss: {:.3f}"
                  .format(current_epoch, train_score, val_score, test_score))
        print("=" * 110)

        epoch_dict = {"epoch": current_epoch,
                    "train_loss": train_score[0] if task == "classification" else train_score,
                    "val_loss": val_score[0] if task == "classification" else val_score,
                    "test_loss": test_score[0] if task == "classification" else test_score,
                    "g_lr": model_general_optimizer.param_groups[0]["lr"],
                    "r_lr": model_response_optimizer.param_groups[0]["lr"]}
        if task == "classification":
            epoch_dict["train_acc"] = train_score[1]
            epoch_dict["val_acc"] = val_score[1]
            epoch_dict["test_acc"] = test_score[1]

        # Gather training data
        train_loss.append(train_score[0] if task == "classification" else train_score)
        val_loss.append(val_score[0] if task == "classification" else val_score)
        test_loss.append(test_score[0] if task == "classification" else test_score)
        if task == "classification":
            val_acc.append(val_score[1])
            test_acc.append(test_score[1])

        # wandb.log(epoch_dict)
        return val_score[0] if task == "classification" else val_score

    def stn_per_epoch_evaluate(current_epoch, train_score = None):
        def evaluate(loader, test = False):
            model.eval()
            correct = total = loss = 0.
            with torch.no_grad():
                for data in loader:
                    features, labels = data[0].to(device), data[1].to(device)
                    repeated_h_tensor = h_container.h_tensor.unsqueeze(0).repeat((features.shape[0], 1))
                    pred = model(features, repeated_h_tensor, repeated_h_tensor)
                    if task == "classification":
                        loss += F.cross_entropy(pred.float(), labels.long(), reduction = "sum").item()
                        hard_pred = torch.max(pred, 1)[1]
                        correct += (hard_pred == labels).sum().item()
                        if not test:
                            all_preds.extend(hard_pred.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    elif task == "regression":
                        loss += F.mse_loss(pred.float(), labels.float(), reduction = "sum").item()
                        if not test:
                            all_preds.extend(pred.float().cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    total += labels.size(0)
            if task == "classification":
                accuracy = correct / float(total)
                return loss / float(total), accuracy
            return loss / float(total)
        

        # if train_score is None:
        train_score = evaluate(train_loader)
        val_score = evaluate(val_loader)
        test_score = evaluate(test_loader, test = True)

        print("=" * 110)
        if task == "classification":
            print("Train Epoch: {} | Train Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}"
                " | Test Loss: {:.3f} | Test Acc: {:.3f}".format(current_epoch, train_score[0], val_score[0],
                                                                val_score[1], test_score[0], test_score[1]))
        else:
            print("Train Epoch: {} | Train Loss: {:.3f} | Val Loss: {:.3f} | Test Loss: {:.3f}"
                  .format(current_epoch, train_score, val_score, test_score))
        print("=" * 110)

        epoch_dict = {"epoch": current_epoch,
                    "train_loss": train_score[0] if task == "classification" else train_score,
                    "val_loss": val_score[0] if task == "classification" else val_score,
                    "test_loss": test_score[0] if task == "classification" else test_score}
        if task == "classification":
            epoch_dict["train_acc"] = train_score[1]
            epoch_dict["val_acc"] = val_score[1]
            epoch_dict["test_acc"] = test_score[1]

        # Gather training data
        train_loss.append(train_score[0] if task == "classification" else train_score)
        val_loss.append(val_score[0] if task == "classification" else val_score)
        test_loss.append(test_score[0] if task == "classification" else test_score)
        if task == "classification":
            val_acc.append(val_score[1])
            test_acc.append(test_score[1])

        # wandb.log(epoch_dict)
        return val_score[0] if task == "classification" else val_score
    
    # --------------------------------------------------------------------------------------------------

    evaluate_fnc = delta_stn_per_epoch_evaluate if delta_stn else stn_per_epoch_evaluate

    stn_trainer = StnTrainer(stn_step_optimizer, train_loader = train_loader, valid_loader = val_loader,
                            test_loader = test_loader, h_container = h_container, evaluate_fnc = evaluate_fnc,
                            device = device, lr_scheduler = None, warmup_epochs = parameters["warmup_epochs"],
                            total_epochs = parameters["total_epochs"], train_steps = parameters["train_steps"],
                            valid_steps = parameters["valid_steps"], log_interval = parameters["log_interval"],
                            patience = None)

    try:
        stn_trainer.train()
        evaluate_fnc(parameters["total_epochs"])
        sys.stdout.flush()

        if task == "classification":
            metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["f1"] = f1_score(all_labels, all_preds, average = "weighted", zero_division = np.nan)
            metrics["precision"] = precision_score(all_labels, all_preds, average = "weighted", zero_division = np.nan)
            metrics["recall"] = recall_score(all_labels, all_preds, average = "weighted", zero_division = np.nan)
        else:
            metrics_reg["mae"] = mean_absolute_error(all_labels, all_preds)
            metrics_reg["mse"] = mean_squared_error(all_labels, all_preds)
            metrics_reg["r2"] = r2_score(all_labels, all_preds)

        # Save results
        if task == "classification":
            results = (metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"])
        else:
            results = (metrics_reg['mae'], metrics_reg['mse'], metrics_reg['r2'])
            
        model_name = "delta_stn" if delta_stn else "stn"
        save_results(dataset, task, results, model_name)
        save_metrics_stn(train_loss, val_loss, test_loss, task, model_name, dataset, val_acc, test_acc)

    except KeyboardInterrupt:
        print("=" * 110)
        print("Exiting from training early ...")
        sys.stdout.flush()

