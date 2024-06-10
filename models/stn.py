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
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout0"), self.training)

        x = self.layers[0](x, h_net)
        x = F.relu(x)
        if "dropout1" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout1"), self.training)

        x = self.layers[1](x, h_net)
        x = F.relu(x)
        if "dropout2" in self.h_container.h_dict:
            x = dropout(x, self.h_container.transform_perturbed_hyper(h_tensor, "dropout2"), self.training)

        x = self.layers[2](x, h_net)
        return x
    
def train_stn_model(input_size, output_size, train_loader, val_loader, test_loader, 
                    hidden_sizes = [64, 64, 64], task = "classification", delta_stn = False):
    # Load parameters from config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        parameters = config["stn"]
    
    # wandb.init(project = f"stn_iris",
    #        tensorboard = True,
    #        dir = f"results/stn_results")

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
    criterion = nn.CrossEntropyLoss(reduction = "mean").to(device)

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
    def delta_stn_per_epoch_evaluate(current_epoch, train_loss = None):
        def evaluate(loader):
            model.eval()
            correct = total = loss = 0.
            with torch.no_grad():
                for data in loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    repeated_h_tensor = model.h_container.h_tensor.unsqueeze(0).repeat((images.shape[0], 1))
                    pred = model(images, repeated_h_tensor - repeated_h_tensor.detach(), repeated_h_tensor)
                    loss += F.cross_entropy(pred.float(), labels.long(), reduction = "sum").item()
                    hard_pred = torch.max(pred, 1)[1]
                    total += labels.size(0)
                    correct += (hard_pred == labels).sum().item()
            accuracy = correct / float(total)
            mean_loss = loss / float(total)
            return mean_loss, accuracy

        if train_loss is None:
            train_loss, train_acc = evaluate(train_loader)
        val_loss, val_acc = evaluate(val_loader)
        tst_loss, tst_acc = evaluate(test_loader)

        print("=" * 80)
        print("Train Epoch: {} | Trn Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}"
            " | Test Loss: {:.3f} | Test Acc: {:.3f}".format(current_epoch, train_loss, val_loss,
                                                            val_acc, tst_loss, tst_acc))
        print("=" * 80)

        epoch_dict = {"epoch": current_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": tst_loss,
                    "test_acc": tst_acc,
                    "g_lr": model_general_optimizer.param_groups[0]["lr"],
                    "r_lr": model_response_optimizer.param_groups[0]["lr"]}

        # wandb.log(epoch_dict)
        return val_loss

    def stn_per_epoch_evaluate(current_epoch, train_loss = None):
        def evaluate(loader):
            model.eval()
            correct = total = loss = 0.
            with torch.no_grad():
                for data in loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    repeated_h_tensor = h_container.h_tensor.unsqueeze(0).repeat((images.shape[0], 1))
                    pred = model(images, repeated_h_tensor, repeated_h_tensor)
                    loss += F.cross_entropy(pred.float(), labels.long(), reduction="sum").item()
                    hard_pred = torch.max(pred, 1)[1]
                    total += labels.size(0)
                    correct += (hard_pred == labels).sum().item()
            accuracy = correct / float(total)
            mean_loss = loss / float(total)
            return mean_loss, accuracy

        if train_loss is None:
            train_loss, train_acc = evaluate(train_loader)
        val_loss, val_acc = evaluate(val_loader)
        tst_loss, tst_acc = evaluate(test_loader)

        print("=" * 80)
        print("Train Epoch: {} | Trn Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}"
            " | Test Loss: {:.3f} | Test Acc: {:.3f}".format(current_epoch, train_loss, val_loss,
                                                            val_acc, tst_loss, tst_acc))
        print("=" * 80)

        epoch_dict = {"epoch": current_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": tst_loss,
                    "test_acc": tst_acc,
                    "lr": model_optimizer.param_groups[0]["lr"]}

        # wandb.log(epoch_dict)
        return val_loss
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

    except KeyboardInterrupt:
        print("=" * 80)
        print("Exiting from training early ...")
        sys.stdout.flush()

