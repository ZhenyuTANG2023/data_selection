import logging
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from typing import Dict
import copy

class EarlyStopping():
    def __init__(self, patience:int, logger:logging.Logger, verbose:bool=True, delta:float=0.):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger
        self.saved_params:Dict[str, torch.Tensor] = None
    
    def __call__(self, val_loss, model:nn.Module, path:str):
        self.logger.info(f"val_loss={val_loss}")
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

            
    def save_checkpoint(self, val_loss, model:nn.Module, path:str):
        if self.verbose:
            self.logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
            self.logger.info(f"Checkpoint will not be saved currently due to settings")
        # Not saving the models currently
        # os.makedirs(path, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(path, "model_checkpoint.pth"))
        self.saved_params = copy.deepcopy(model.state_dict())

        self.val_loss_min = val_loss
        
        
def run_validation(logger:logging.Logger, model:nn.Module, dataloader:DataLoader, device:str, multi_label:bool, criterion, phase:str, num_classes:int, valid_loss:list=None):
    model.eval()
    num_correct = np.zeros((num_classes,), dtype=np.float32) if multi_label else 0
    num_samples = 0
    test_loss = 0.
    valid_epoch_loss = []
    y_true = []
    if multi_label:
        y_score = []
    else:
        y_pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(torch.float32).to(device)
            y:torch.Tensor = y.to(torch.float32 if multi_label else torch.long).to(device)
            y_true.append(y.cpu().detach().numpy())
            scores:torch.Tensor = model(x)
            if multi_label:
                scores = torch.sigmoid(scores)
                num_correct += np.sum(((scores > 0.5) == (y > 0)).cpu().detach().numpy(), axis=0)
                y_score.append(scores.cpu().detach().numpy())
            else:
                _, predictions = scores.max(1)
                y_pred.append(predictions.cpu().detach().numpy())
                num_correct += (predictions == y).sum()
            num_samples += y.shape[0]
            tmp_test_loss = criterion(scores, y).item() * len(x)
            test_loss += tmp_test_loss
            valid_epoch_loss.append(tmp_test_loss)
            if valid_loss:
                valid_loss.append(test_loss)
    for i in range(len(valid_epoch_loss)):
        valid_epoch_loss[i] /= num_samples
    y_true = np.concatenate(y_true)
    if multi_label:
        # for multi label, output AUC score
        y_score = np.concatenate(y_score)
        auc = roc_auc_score(y_true, y_score, average=None)
        logger.info(f"{phase} AUC: {auc}")
        logger.info(f"average AUC: {np.mean(auc)}")
        logger.info(f"{phase} accuracy: {num_correct} / {num_samples} \n\t\t= {np.round(num_correct / num_samples * 100, 4)}")
    else:
        y_pred = np.concatenate(y_pred)
        logger.info(f"{phase} accuracy: {num_correct} / {num_samples} = {float(num_correct) / num_samples * 100:.4f}%")
        logger.info(f"{phase} balanced accuracy: {balanced_accuracy_score(y_true, y_pred)}")
        logger.info(f"{phase} f1 score micro: {f1_score(y_true, y_pred, average='micro')}")
        logger.info(f"{phase} f1 score macro: {f1_score(y_true, y_pred, average='macro')}")
        logger.info(f"{phase} f1 score weighted: {f1_score(y_true, y_pred, average='weighted')}")
    logger.info(f"{phase} loss: {test_loss} / {num_samples} = {float(test_loss) / num_samples:.4f}")
    return valid_epoch_loss


def visualize_training(
    logger:logging.Logger, 
    train_loss:list, 
    train_epochs_loss:list, 
    valid_epochs_loss:list, 
    timestamp:int
):
    logger.info("Visualizing the training curve")
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], "-o", label="train_loss")
    if valid_epochs_loss is not None:
        plt.plot(valid_epochs_loss[1:], "-o", label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    os.makedirs("./img/fullset", exist_ok=True)
    plt.savefig(f"./img/fullset/{timestamp}.png")


def set_bn_eval(module:nn.Module):
    """
        Setting a model to eval mode if it is a Batch Norm. Usually used with model.apply()

        Params:
        ----------
            module:     a differentiable module
        
        Returns:
        ----------
            None
    """
    classname = module.__class__.__name__
    if classname.find('BatchNorm') != -1:
        module.eval()

def set_bn_train(module:nn.Module):
    """
        Setting a model to train mode if it is a Batch Norm. Usually used with model.apply()

        Params:
        ----------
            module:     a differentiable module
        
        Returns:
        ----------
            None
    """
    classname = module.__class__.__name__
    if classname.find('BatchNorm') != -1:
        module.train()

def retrieve_classes(dataset:Dataset):
    classes = np.unique(dataset.classes if "classes" in dir(dataset) else dataset.labels)
    num_classes = len(classes)
    return num_classes, classes
