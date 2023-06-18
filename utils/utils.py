import random
from typing import List, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def seed_everything(seed:int=123):
    """
        Set the random seeds for the experiment. All random seeds in various packages are set the same.
        Note: in some algorithms there are seeds to set when calling the functions, which is not taken into
              consideration here.

        Params:
        ----------
        seed        the random seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

def zero_state_dict(si:Dict[str, torch.Tensor]):
    """
        Set state dict to zeros.

        Params:
        ----------
        si          state dict
    """
    with torch.no_grad():
        for _k in si.keys():
            si[_k].zero_()


def clone_weights(model:nn.Module):
    ret:Dict[str, torch.Tensor] = {}
    for _k in model.state_dict().keys():
        ret[_k] = model.state_dict()[_k].clone().detach()
    return ret


def nth_largest(a:np.ndarray, n:int):
    return np.partition(a.flatten(), -n)[-n]


def arg_nth_largest(a:np.ndarray, n:int):
    return np.argpartition(a.flatten(), -n)[-n]


class TorchHook:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self._features_out_hook:List[torch.Tensor] = []
        self._grads_out_hook:List = []
    
    def forward_hook(self, module:nn.Module, fea_in:torch.Tensor, fea_out:torch.Tensor):
        self._features_out_hook.append(fea_out.clone().detach())
        return None
    
    def backward_hook(self, module:nn.Module, grad_input, grad_output):
        self._grads_out_hook.append(grad_output.clone())

    def register_hook(self, model:nn.Module, forward_layer_names:List[str], backward_layer_names:List[str]):
        for (name, module) in model.named_modules():
            if name in forward_layer_names:
                module.register_forward_hook(hook=self.forward_hook)
            if name in backward_layer_names:
                module.register_full_backward_hook(hook=self.backward_hook)

    @property
    def grads_out_hook(self):
        return self._grads_out_hook

    @property
    def features_out_hook(self):
        return self._features_out_hook


