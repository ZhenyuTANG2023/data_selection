import logging
from typing import List, Dict, Any, Tuple
import torch.nn as nn
from .estimate_updates import adam_estimate_update, sgd_estimate_update
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from utils.training import set_bn_eval, set_bn_train
import time
from torch.nn.functional import one_hot
import numpy as np
from .observation import ObsDataManager
from sklearn.cluster import KMeans
import random
import numpy as np

def get_group_name(feat_name:str, feat_dims:str):
    """obtain the feature group name

    Args:
        feat_name (str): the name for a feature
        feat_dims (str): the number of dimensions, either "1-dim" or "n-dim"

    Returns:
        str: the feature group name
    """
    if feat_dims not in ["1-dim", "n-dim"]:
        raise NotImplementedError(f"feat dims {feat_dims} is not supported")
    if feat_dims == "n-dim":
        return feat_name.split(".")[0]
    if feat_dims != "1-dim":
        raise NotImplementedError
    return "all"


def group_feat(feats:np.ndarray, feat_names:List[str], feat_nelements:np.ndarray, feat_dims:str):
    """group the features

    Args:
        feats (np.ndarray): original features
        feat_names (List[str]): the names for the features
        feat_nelements (np.ndarray): the number of elements for each feature
        feat_dims (str): the number of group dimensions, either "1-dim" or "n-dim"

    Returns:
        feat_groups (np.ndarray): groupped features
        feat_group_names (List[str]): the names for the groupped features
        feat_group_nelements (np.ndarray): the number of elements for each groupped feature
    """
    feat_group_names:List[str] = []

    for feat_name in feat_names:
        feat_group_name = get_group_name(feat_name, feat_dims)
        if feat_group_name not in feat_group_names:
            feat_group_names.append(feat_group_name)
    
    feat_groups:np.ndarray = np.zeros((feats.shape[0], len(feat_group_names)), dtype=np.float64)
    feat_group_nelements:np.ndarray = np.zeros((feats.shape[0],), dtype=int)

    for dim, feat_name in zip(range(feats.shape[1]), feat_names):
        feat_group_name = get_group_name(feat_name, feat_dims)
        feat_group_idx:int = feat_group_names.index(feat_group_name)
        feat_group_nelements[feat_group_idx] += feat_nelements[dim]
        feat_groups[:, feat_group_idx] += feats[:, dim] * feat_nelements[dim]

    for feat_group_idx in range(len(feat_group_names)):
        feat_groups[:, feat_group_idx] /= feat_group_nelements[feat_group_idx]
    
    return feat_groups, feat_group_names, feat_group_nelements


class DataEvaluator:
    def __init__(
        self,
        eval_method:str,
        data_val_feat:str,
        multi_label:bool, 
        valuation_batch_size:int,
        device:str,
        params_to_update_names:List[str],
        run_with_slurm:str,
        logger:logging.Logger
    ) -> None:
        """data evaluator class

        Args:
            eval_method (str): _description_
            data_val_feat (str): _description_
            multi_label (bool): _description_
            valuation_batch_size (int): _description_
            device (str): _description_
            params_to_update_names (List[str]): _description_
            run_with_slurm (str): _description_
            logger (logging.Logger): _description_
        """
        self.valuation_methods = {
            "random": self.random_value,
            "balanced_random": self.balanced_random_value,
            "data_si": self.data_si,
            "data_si_approx": self.data_si_approx,
            "gradnorm_approx": self.gradnorm_approx,
            "gradnorm": self.gradnorm,
            "smallest_margin": self.smallest_margin_value,
            "least_confidence": self.least_confidence_value
        }
        if eval_method not in self.valuation_methods:
            raise NotImplementedError(f"data valuation method `{eval_method}` is not supported; supported methods: {list(self.valuation_methods.keys())}")
        self._eval_method_name = eval_method
        self._val_func = self.valuation_methods[eval_method]
        if self._val_func is None:
            raise NotImplementedError(f"data valuation method `{eval_method}` is not temporarily available and we are working on it.")
        self.data_val_feat = data_val_feat
        self._multi_label = multi_label
        self._val_time:float = -1
        self._batch_size = valuation_batch_size
        self._device = device
        self._params_to_update_names = params_to_update_names
        self._run_with_slurm = run_with_slurm
        self._logger = logger

        self._buffer = None
        self._coreset = True


    def _save_raw_dvs(self, result:np.ndarray, feat_name_nelement:List[Tuple[str, int]], folder:str, epoch:int):
        """save the raw data values locally

        Args:
            result (np.ndarray): raw data values
            feat_name_nelement (List[Tuple[str, int]]): the name and number of element for each feature
            folder (str): the folder for saving the values
            epoch (int): the epoch number
        """
        ObsDataManager.save_feat(epoch, folder, result)
        ObsDataManager.save_feat_info(epoch, folder, feat_name_nelement)
        
    
    def data_value(
        self, 
        indices:list, 
        model:nn.Module, 
        dataset:Dataset, 
        num_classes:int, 
        optimizer:torch.optim.Optimizer, 
        criterion:nn.modules.loss._Loss,
        params_to_update:List[torch.Tensor],
        epoch:int,
        folder:str,
    ):
        """evaluate data value

        Args:
            indices (list): the training indices
            model (nn.Module): current model status
            dataset (Dataset): the training dataset
            num_classes (int): the number of classes in this task
            optimizer (torch.optim.Optimizer): current optimizer status
            criterion (nn.modules.loss._Loss): the criterion used in training
            params_to_update (List[torch.Tensor]): the parameters to update
            epoch (int): the epoch number 
            folder (str): the folder to save the results

        Returns:
            feat_groups (np.ndarray): groupped data features
            feat_group_names (List[str]): the names for the groupped data features
            feat_group_nelements (np.ndarray): the number of elements for each groupped data feature
        """
        result:torch.Tensor = self._val_func(indices, model, dataset, num_classes, optimizer, criterion)
        self._logger.info(f"data evaluation overhead: {self.valuation_time:.3f} s")
        
        if self._eval_method_name in ["random", "balanced_random", "smallest_margin", "least_confidence"]:
            return result, ["all"], 1
        
        feat_name_nelement:List[Tuple[str, int]] = []
        for (name, param) in zip(self._params_to_update_names, params_to_update):
            feat_name_nelement.append((name, param.nelement()))
        feat_names = [item[0] for item in feat_name_nelement]
        feat_nelements = [item[1] for item in feat_name_nelement]

        feat_groups, feat_group_names, feat_group_nelements = group_feat(
            result.clone().detach().cpu().numpy(), feat_names, feat_nelements, self.data_val_feat)
        self._save_raw_dvs(feat_groups, feat_group_nelements, folder, epoch)
        return feat_groups, feat_group_names, feat_group_nelements

    @property
    def valuation_time(self) -> float:
        """obtain the valuation overhead

        Returns:
            float: the valuation overhead
        """
        return self._val_time
    
    
    def _forward_backward_gradient_accumulation(self, model:nn.Module, dataset:Dataset, criterion:nn.modules.loss._Loss):
        """to perform gradient accumulation

        Args:
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            criterion (nn.modules.loss._Loss): the criterion used
        """
        num_data_points = len(dataset)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=1)
        dataloader = dataloader if self._run_with_slurm else tqdm(dataloader, desc="gradient accumulation")
        for idx, (x, y) in enumerate(dataloader):
            x:torch.Tensor = x.to(torch.float32).to(self._device)
            y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
            scores = model(x)
            # gradient accumulation
            if self._multi_label:
                scores = torch.sigmoid(scores)
            if len(y.shape) == len(x.shape): y = torch.squeeze(y, dim=1)
            loss:torch.Tensor = criterion(scores, y) * (x.shape[0] / num_data_points)
            # remove computation graphs
            loss.backward()
            
    
    def _compute_delta(self, param_group:dict, param:torch.Tensor, gradients:torch.Tensor, param_state:Dict[str, Any], optimizer:torch.optim.Optimizer) -> torch.Tensor:
        """optimizer-specific estimator of delta_weight

        Args:
            param_group (dict): a specific param group in the optimizer
            param (torch.Tensor): the param value of the param group
            gradients (torch.Tensor): the gradients of the param group
            param_state (Dict[str, Any]): the param state of the param group
            optimizer (torch.optim.Optimizer): the optimizer used

        Returns:
            delta_weights (torch.Tensor): the computed delta weights
        """
        if isinstance(optimizer, torch.optim.SGD):
            buffer = param_state["momentum_buffer"] if "momentum_buffer" in param_state.keys() else None
            weight_decay = param_group['weight_decay']
            momentum = param_group['momentum']
            lr = param_group['lr']
            dampening = param_group['dampening']
            nesterov = param_group['nesterov']
            maximize = param_group['maximize']
            foreach = param_group['foreach']
            if foreach:
                raise NotImplementedError
            return sgd_estimate_update(lr, momentum, buffer, dampening, gradients, weight_decay, param, nesterov, maximize)
        elif isinstance(optimizer, torch.optim.Adam):
            fused = param_group["fused"]
            beta1, beta2 = param_group["betas"]
            amsgrad = param_group["amsgrad"]
            lr = param_group["lr"]
            weight_decay = param_group["weight_decay"]
            eps = param_group["eps"]
            maximize = param_group["maximize"]
            foreach = param_group["foreach"]
            capturable = param_group["capturable"]
            differentiable = param_group["differentiable"]
            if param_state is None or len(param_state) == 0:
                exp_avg = torch.zeros_like(param)
                exp_avg_sq = torch.zeros_like(param)
                max_exp_avg_sq = torch.zeros_like(param)
                state_step = torch.tensor(0.)
            else:
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]
                max_exp_avg_sq = param_state["max_exp_avg_sq"] if amsgrad else None
                state_step:torch.Tensor = param_state["step"]
            if foreach or fused or capturable:
                raise NotImplementedError
            return adam_estimate_update(lr, gradients, weight_decay, param, maximize, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, amsgrad, beta1, beta2, eps, capturable, differentiable)
        else:
            raise NotImplementedError(f"Optimizer named {optimizer.__class__.__name__}: not implemented yet")


    def _compute_delta_weights(self, optimizer:torch.optim.Optimizer) -> Dict[int, torch.Tensor]:
        """computes the delta weights of a gradient accumulated model optimizer

        Args:
            optimizer (torch.optim.Optimizer): the optimizer used

        Returns:
            delta_weights (Dict[int, torch.Tensor]): the delta weights estimated
        """
        if len(optimizer.param_groups) > 1:
            print(len(optimizer.param_groups))
            raise NotImplementedError("The current implementation supports single param group only")
        param_group = optimizer.param_groups[0]
        delta_weights:Dict[int, torch.Tensor] = {}
        for idx, param_name in enumerate(self._params_to_update_names):
            param:torch.Tensor = param_group["params"][idx]
            delta_weights[idx] = self._compute_delta(
                param_group, 
                param, 
                param.grad, 
                optimizer.state[param],
                optimizer
            )
        
        return delta_weights
    
    
    @torch.no_grad()
    def _compute_delta_times_grad_no_grad(self, delta_tensor:torch.Tensor, param_tensor:torch.Tensor):
        return delta_tensor * param_tensor


    def _compute_delta_times_grad(self, delta:Dict[int, torch.Tensor], optimizer:torch.optim.Optimizer):
        if len(optimizer.param_groups) > 1:
            raise NotImplementedError("The current implementation supports single param group only")
        param_group = optimizer.param_groups[0]
        return {k: self._compute_delta_times_grad_no_grad(delta[k], param_group["params"][k].grad) for k in delta}
    
    
    @torch.no_grad()
    def _compute_grad_norm_no_grad(self, param_tensor:torch.Tensor):
        # return torch.norm(param_tensor)
        return torch.sum(torch.square(param_tensor))
    
    
    def _compute_grad_norm(self, optimizer:torch.optim.Optimizer, full:bool=False):
        if len(optimizer.param_groups) > 1:
            raise NotImplementedError("The current implementation supports single param group only")
        param_group = optimizer.param_groups[0]
        if full:
            raise NotImplementedError
        return {k: self._compute_grad_norm_no_grad(param_group["params"][k].grad) if param_group["params"][k].grad is not None else None for k in range(len(param_group["params"]))}
    
    
    def gradnorm_approx(self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, criterion:nn.modules.loss._Loss):
        """computes approximate GradNorm values

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the approximate GradNorm values
        """
        _start_time = time.time()
        loss_changes = torch.zeros((len(indices), 1), dtype=torch.float64, requires_grad=False)
        subset = Subset(dataset, indices)

        # set BNs eval
        model.apply(set_bn_eval)

        # batch forward and approximate via dy * grad_y
        dataloader = DataLoader(subset, batch_size=self._batch_size, shuffle=False, num_workers=24)
        dataloader_iter = dataloader if self._run_with_slurm else tqdm(dataloader, desc="approx si computation")
        for idx, (x, y) in enumerate(dataloader_iter):
            x:torch.Tensor = x.to(torch.float32).to(self._device)
            y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
            scores = model(x)
            if self._multi_label:
                scores = torch.sigmoid(scores)
            if len(y.shape) == len(x.shape): y = torch.squeeze(y, dim=1)
            loss:torch.Tensor = criterion(scores, y)
            grad_scores = torch.autograd.grad(loss, scores)[0]
            grad_y = grad_scores.clone().detach()
            with torch.no_grad():
                loss_changes[idx * self._batch_size : (idx + 1) * self._batch_size] = torch.sum(torch.square(grad_y), dim=list(range(len(grad_y.shape)))[1:]).reshape(-1, 1)
            optimizer.zero_grad()

        model.apply(set_bn_train)
        
        self._val_time = time.time() - _start_time

        return loss_changes
    
    def data_si_approx(self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, criterion:nn.modules.loss._Loss):
        """computes approximate Data SI values

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the approximate Data SI values
        """
        _start_time = time.time()
        loss_changes = torch.zeros((len(indices), 1), dtype=torch.float64, requires_grad=False)
        subset = Subset(dataset, indices)

        # set BNs eval
        model.apply(set_bn_eval)

        # batch forward and approximate via dy * grad_y
        dataloader = DataLoader(subset, batch_size=self._batch_size, shuffle=False, num_workers=24)
        dataloader_iter = dataloader if self._run_with_slurm else tqdm(dataloader, desc="approx si computation")
        for idx, (x, y) in enumerate(dataloader_iter):
            x:torch.Tensor = x.to(torch.float32).to(self._device)
            y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
            # y_onehot:torch.Tensor = torch.zeros((y.shape[0], num_classes)).to(self._device)
            y_onehot:torch.Tensor = one_hot(y, num_classes=num_classes)
            scores = model(x)
            if self._multi_label:
                scores = torch.sigmoid(scores)
            if len(y.shape) == len(x.shape): y = torch.squeeze(y, dim=1)
            loss:torch.Tensor = criterion(scores, y)
            grad_scores = torch.autograd.grad(loss, scores)[0]
            grad_y = grad_scores.clone().detach()
            with torch.no_grad():
                loss_changes[idx * self._batch_size : (idx + 1) * self._batch_size] = torch.mean(grad_y * (y_onehot - scores), dim=1).reshape(-1, 1)
            optimizer.zero_grad()

        model.apply(set_bn_train)
        
        self._val_time = time.time() - _start_time

        return loss_changes
    
    
    def data_si(self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, criterion:nn.modules.loss._Loss):
        """computes Data SI values

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the Data SI values
        """
        _start_time = time.time()
        loss_changes = torch.zeros((len(indices), len(self._params_to_update_names)), dtype=torch.float64, requires_grad=False)
        subset = Subset(dataset, indices)
        
        # set BNs eval
        model.apply(set_bn_eval)

        # accumulate the data and obtain the parameter-wise weights
        self._forward_backward_gradient_accumulation(model, subset, criterion)
        delta_weights = self._compute_delta_weights(optimizer)
        optimizer.zero_grad()

        dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
        dataloader_iter = dataloader if self._run_with_slurm else tqdm(dataloader, desc="si computation")
        for idx, (x, y) in enumerate(dataloader_iter):
            x:torch.Tensor = x.to(torch.float32).to(self._device)
            y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
            scores = model(x)
            if self._multi_label:
                scores = torch.sigmoid(scores)
            if len(y.shape) == len(x.shape): y = torch.squeeze(y, dim=1)
            loss:torch.Tensor = criterion(scores, y)
            loss.backward()
            dtg_dict = self._compute_delta_times_grad(delta_weights, optimizer)
            loss_changes[idx] = torch.Tensor([torch.mean(dtg_dict[param_idx]) for param_idx in range(len(self._params_to_update_names))])
            optimizer.zero_grad()

        model.apply(set_bn_train)
        
        self._val_time = time.time() - _start_time

        return loss_changes
    
    
    def gradnorm(self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, criterion:nn.modules.loss._Loss):
        """computes GradNorm values

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the GradNorm values
        """
        _start_time = time.time()
        
        grad_norms = torch.zeros((len(indices), len(self._params_to_update_names)), dtype=torch.float64, requires_grad=False)
        subset = Subset(dataset, indices)
        
        model.apply(set_bn_eval)
        
        dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
        dataloader_iter = dataloader if self._run_with_slurm else tqdm(dataloader, desc="grad norm computation")
        
        for idx, (x, y) in enumerate(dataloader_iter):
            x:torch.Tensor = x.to(torch.float32).to(self._device)
            y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
            scores = model(x)
            if self._multi_label:
                scores = torch.sigmoid(scores)
            if len(y.shape) == len(x.shape): y = torch.squeeze(y, dim=1)
            loss:torch.Tensor = criterion(scores, y)
            loss.backward()
            dtg_dict = self._compute_grad_norm(optimizer)
            grad_norms[idx] = torch.Tensor([dtg_dict[param_idx] if dtg_dict[param_idx] is not None else -1 for param_idx in range(len(self._params_to_update_names)) ])
            optimizer.zero_grad()
        
        model.apply(set_bn_train)
        
        self._val_time = time.time() - _start_time
        
        return grad_norms
    
    
    def smallest_margin_value(
        self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, 
        criterion:nn.modules.loss._Loss
    ):
        """computes margin values

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the margin values
        """
        neg_margin_value = torch.zeros((len(indices), 1), dtype=torch.float64, requires_grad=False)
        subset = Subset(dataset, indices)
        
        model.apply(set_bn_eval)
        
        dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
        dataloader_iter = dataloader if self._run_with_slurm else tqdm(dataloader, desc="margin computation")
        
        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader_iter):
                x:torch.Tensor = x.to(torch.float32).to(self._device)
                y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
                scores = model(x)
                if self._multi_label:
                    raise NotImplementedError("Margin based method is not implemented on multi label tasks yet")
                indices = torch.argsort(scores, axis=1)[0, -2:]
                neg_margin_value[idx, 0] = scores[0, indices[0]] - scores[0, indices[1]]
        
        model.apply(set_bn_train)
        
        self._val_time = 0
        
        return neg_margin_value
    
    
    def least_confidence_value(
        self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, 
        criterion:nn.modules.loss._Loss
    ):
        """computes negative confidence values

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the negative confidence values
        """
        neg_margin_value = torch.zeros((len(indices), 1), dtype=torch.float64, requires_grad=False)
        subset = Subset(dataset, indices)
        
        model.apply(set_bn_eval)
        
        dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
        dataloader_iter = dataloader if self._run_with_slurm else tqdm(dataloader, desc="confidence computation")
        
        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader_iter):
                x:torch.Tensor = x.to(torch.float32).to(self._device)
                y:torch.Tensor = y.to(torch.float32 if self._multi_label else torch.long).to(self._device)
                scores = model(x)
                if self._multi_label:
                    raise NotImplementedError("Margin based method is not implemented on multi label tasks yet")
                indices = torch.argsort(scores, axis=1)[0, -1]
                neg_margin_value[idx, 0] = -scores[0, indices]
        
        model.apply(set_bn_train)
        
        self._val_time = 0
        
        return neg_margin_value


    def random_value(
        self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer, 
        criterion:nn.modules.loss._Loss
    ):
        """random value selection, apply values so that the top_ratio strategy selects samples randomly

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the random values
        """
        self._val_time = 0.
        if not self._coreset or self._buffer is None:
            self._buffer = torch.rand((len(indices), 1), dtype=torch.float64, requires_grad=False)
        return self._buffer
    
    
    def balanced_random_value(
        self, indices:list, model:nn.Module, dataset:Dataset, num_classes:int, optimizer:torch.optim.Optimizer,
        criterion:nn.modules.loss._Loss
    ):
        """balanced random value selection, apply values so that the top_ratio strategy selects balanced randomly

        Args:
            indices (list): the training indices
            model (nn.Module): the current model status
            dataset (Dataset): the dataset used
            num_classes (int): the number of classes
            optimizer (torch.optim.Optimizer): the optimizer used
            criterion (nn.modules.loss._Loss): the criterion used

        Returns:
            value: the balanced random values
        """
        self._val_time = 0.
        if not self._coreset or self._buffer is None:
            dataset_size = len(indices)
            self._buffer = torch.zeros((len(indices), 1), dtype=torch.float64, requires_grad=False)
            classes_array = np.zeros((dataset_size), dtype=int)
            subset = Subset(dataset, indices)
            range_iter = range(dataset_size) if self._run_with_slurm else trange(dataset_size)
            for idx in range_iter:
                x, y = subset[idx]
                classes_array[idx] = y
            range_iter = range(num_classes) if self._run_with_slurm else trange(num_classes, desc="balanced rating")
            for cls_idx in range_iter:
                cls_mask = classes_array == cls_idx
                randperm = np.random.permutation(np.where(cls_mask))
                rk = dataset_size
                for idx in randperm:
                    self._buffer[idx, 0] = rk
                    rk -= 1
        return self._buffer

    

class DataSelector:
    def __init__(self, strategy:str, sample_size:float, lr_adjust:bool, optimizer:torch.optim.Optimizer, logger:logging.Logger) -> None:
        """data selector class

        Args:
            strategy (str): the name of selection strategy
            sample_size (float): the size (times full set) to be sampled
            lr_adjust (bool): toggles using learning rate adjustment
            optimizer (torch.optim.Optimizer): the optimizer used
            logger (logging.Logger): the logger
        """
        self._lrate_adjust:bool = lr_adjust
        self._selection_strategies = {
            "thresh": self.thresholding,
            "cluster": self.clustering,
            "topratio": self.top_ratio,
        }
        
        self._sel_ratio:float = 1.0
        self._sample_size = sample_size
        self._strategy = strategy
        self._selection_func = self._selection_strategies[strategy]
        self._optimizer = optimizer
        self._logger = logger
        
    def select_data(self, feat_groups:np.ndarray, dataset:Dataset, batch_size:int):
        """select data

        Args:
            feat_groups (np.ndarray): the groupped feature
            dataset (Dataset): the dataset to be selected from
            batch_size (int): the batch size for returned dataloader

        Returns:
            sel_dataloader: the selected dataloader
        """
        sel_dataloader, sel_ratio = self._selection_func(feat_groups, dataset, batch_size)
        self._sel_ratio = sel_ratio
        self.lrate_adjust()
        return sel_dataloader
        
    def lrate_deadjust(self):
        """adjust the learning rate back for learning rate scheduling
        """
        if self._lrate_adjust and self._sel_ratio < 1.0:
            for g in self._optimizer.param_groups:
                g["lr"] = g["lr"] / self._sel_ratio
            self._sel_ratio = 1.0
            
    def lrate_adjust(self):
        """adjust learning rate
        """
        if self._lrate_adjust and self._sel_ratio < 1.0:
            for g in self._optimizer.param_groups:
                g["lr"] = g["lr"] * self._sel_ratio
    
    def top_ratio(self, feat_groups:np.ndarray, dataset:Dataset, batch_size:int):
        """top ratio strategy, selecting the top-k samples ordered by a certain score

        Args:
            feat_groups (np.ndarray): the groupped feature
            dataset (Dataset): the dataset to be selected from
            batch_size (int): the batch size for returned dataloader

        Returns:
            sel_dataloader: the selected dataloader
        """
        if feat_groups.shape[1] > 1:
            raise NotImplementedError
        try:
            top_indices = np.argsort(feat_groups.reshape(-1,))[::-1][:int(len(dataset) * self._sample_size)]
        except:
            top_indices = torch.argsort(feat_groups.reshape(-1,), descending=True)[:int(len(dataset) * self._sample_size)]
            
        subset = Subset(dataset, top_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=10), self._sample_size
    
    def clustering(self, feat_groups:np.ndarray, dataset:Dataset, batch_size:int):
        """clustering strategy, discard the largest cluster if it exceeds certain amount (50% of full set here)

        Args:
            feat_groups (np.ndarray): the groupped feature
            dataset (Dataset): the dataset to be selected from
            batch_size (int): the batch size for returned dataloader

        Returns:
            sel_dataloader: the selected dataloader
        """
        # params
        num_clusters = 10
        stability_sample_ratio = 0.0
        max_cluster_thresh = 0.5
        
        # cluster and count
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(feat_groups)
        cluster_labels:List[int] = kmeans.labels_.tolist()
        cluster_centers:List[np.ndarray] = kmeans.cluster_centers_.tolist()
        count_arr = np.zeros((num_clusters,), dtype=int)
        for i in range(num_clusters):
            count_arr[i] = cluster_labels.count(i)
        self._logger.info(f"cluster ratios: {count_arr / np.sum(count_arr)}")
        self._logger.info(f"corresponding centers: {cluster_centers}")
        
        if np.max(count_arr) / np.sum(count_arr) > max_cluster_thresh:
            idx_mask = np.array(cluster_labels) != np.argmax(count_arr)
            tail_indices = np.where(idx_mask)[0]
            center_indices:np.ndarray = np.where(~idx_mask)[0]
            if stability_sample_ratio > 0.:
                subset_center_indices = random.sample(center_indices.tolist(), int(len(center_indices) * stability_sample_ratio))
                subset = Subset(dataset, np.concatenate([tail_indices, subset_center_indices]))
            else:
                subset_center_indices = []
                subset = Subset(dataset, tail_indices)
            self._logger.info(f"Selected ratio: {len(tail_indices)}/{len(dataset)}={len(tail_indices)/len(dataset)*100:.2f}%; subset size: {len(np.concatenate([tail_indices, subset_center_indices]))}/{len(dataset)}")
            return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=10), (len(subset_center_indices) + len(tail_indices)) / len(dataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10), 1.0
    
    def thresholding(self, feat_groups:np.ndarray, dataset:Dataset, batch_size:int):
        """group the features by thresholds, select the data with value between 0.1 * scale and 50 * scale

        Args:
            feat_groups (np.ndarray): the groupped feature
            dataset (Dataset): the dataset to be selected from
            batch_size (int): the batch size for returned dataloader

        Returns:
            sel_dataloader: the selected dataloader
        """
        if feat_groups.shape[1] > 1:
            raise NotImplementedError
        
        # params
        threshold_lower = 0.1
        threshold_upper = 50
        
        scale = np.mean(feat_groups, axis=0)
        feat_rescaled = feat_groups / scale
        feat_norm = np.linalg.norm(feat_rescaled, axis=1)
        lower_mask = feat_norm > threshold_lower
        upper_mask = feat_norm < threshold_upper
        mask = lower_mask & upper_mask
        self._logger.info(f"Scale: {scale}")
        self._logger.info(f"Lower mask: {np.sum(~lower_mask)}/{feat_groups.shape[0]}={np.sum(~lower_mask)/feat_groups.shape[0]*100:.2f}%, mean: {np.mean(feat_groups[~lower_mask], axis=0)}")
        self._logger.info(f"Middle mask: {np.sum(mask)}/{feat_groups.shape[0]}={np.sum(mask)/feat_groups.shape[0]*100:.2f}%, mean: {np.mean(feat_groups[mask], axis=0) if np.sum(mask) > 0 else None}")
        self._logger.info(f"Upper mask: {np.sum(~upper_mask)}/{feat_groups.shape[0]}={np.sum(~upper_mask)/feat_groups.shape[0]*100:.2f}%, mean: {np.mean(feat_groups[~upper_mask], axis=0) if np.sum(~upper_mask) > 0 else None}")
        lower_indices = np.where(~lower_mask)[0]
        upper_indices = np.where(~upper_mask)[0]
        center_indices:np.ndarray = np.where(mask)[0]
        subset = Subset(dataset, center_indices.tolist())
        self._logger.info(f"Selected")
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=10), len(center_indices) / len(dataset)
        