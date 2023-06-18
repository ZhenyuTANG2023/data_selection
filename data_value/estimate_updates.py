import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import math

# test passed
@torch.no_grad()
def sgd_estimate_update(
    lr:float,
    momentum:float,
    momentum_buffer:torch.Tensor,
    dampening:float,
    gradients:torch.Tensor,
    weight_decay:float,
    param:torch.Tensor,
    nesterov:bool,
    maximize:bool
):
    if nesterov or maximize:
        raise NotImplementedError
    if momentum_buffer is None:
        return -lr * (gradients + weight_decay * param)
    return -lr * (momentum * momentum_buffer + (1 - dampening) * (gradients + weight_decay * param))


@torch.no_grad()
def adam_estimate_update(
    lr:float,
    gradients:torch.Tensor,
    weight_decay:float,
    param:torch.Tensor,
    maximize:bool,
    exp_avg:torch.Tensor,
    exp_avg_sq:torch.Tensor,
    max_exp_avg_sq:torch.Tensor,
    step_t:torch.Tensor,
    amsgrad:bool,
    beta1:float,
    beta2:float,
    eps:float,
    capturable:bool,
    differentiable:bool
):
    if maximize:
        raise NotImplementedError
    step = step_t + 1

    grad_dec = torch.add(gradients, param, alpha=weight_decay) if weight_decay != 0 else gradients
    exp_avg_dec = torch.mul(exp_avg, beta1).add_(grad_dec, alpha=1 - beta1)
    exp_avg_sq_dec = torch.mul(exp_avg_sq, beta2).addcmul_(grad_dec, grad_dec.conj(), value=1 - beta2)

    if capturable or differentiable:
        bias_correction1 = 1 - torch.pow(beta1, step)
        biad_correction2 = 1 - torch.pow(beta2, step)

        step_size = lr / bias_correction1
        step_size_neg = step_size.neg()

        biad_correction2_sqrt = biad_correction2.sqrt()
        if amsgrad:
            max_exp_avg_sq_i = torch.maximum(max_exp_avg_sq, exp_avg_sq_dec)
            denom = (max_exp_avg_sq_i.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
        else:
            denom = (exp_avg_sq_dec.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
        return exp_avg_dec / denom
    else:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            max_exp_avg_sq_i = torch.maximum(max_exp_avg_sq, exp_avg_sq_dec)
            denom = (max_exp_avg_sq_i.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq_dec.sqrt() / bias_correction2_sqrt).add_(eps)
        return exp_avg_dec / denom * (-step_size)
    


def seed_everything(seed:int=123):
    """
        Set the random seeds for the experiment. All random seeds in various packages are set the same.
        Note: in some algorithms there are seeds to set when calling the functions, which is not taken into
              consideration here.

        Params:
        ----------
        seed:       the random seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        #TODO: not known now whether there are uncertainty if benchmark is enabled
        cudnn.benchmark = False

def __test_sgd_estimates__():
    import torchvision.models as models
    print("testing SGD estimation")
    seed_everything(1234)
    num_classes = 10
    batch_size = 128
    idx = 0

    import torchvision.models as models
    model:models.ResNet = models.resnet18()
    model.fc = nn.Linear(512, num_classes)
    
    # generate random input and outputs
    data_x = torch.randn(batch_size, 3, 224, 224)
    data_y = torch.randint(0, 10, (batch_size,)).to(torch.long)
    
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # test 1: gradient accumulation in a batch
    optimizer.zero_grad()
    scores = model(data_x)
    loss:torch.Tensor = criterion(scores, data_y)
    loss.backward()

    param_group = optimizer.param_groups[0]
    param:torch.Tensor = param_group["params"][idx]
    buffer = optimizer.state[param]["momentum_buffer"] if "momentum_buffer" in optimizer.state[param].keys() else None
    gradients = param.grad
    weight_decay=param_group['weight_decay']
    momentum=param_group['momentum']
    lr=param_group['lr']
    dampening=param_group['dampening']
    nesterov=param_group['nesterov']
    maximize=param_group['maximize']

    print("===================================================================")
    print("param")
    print(param[0, 0, 0, 0])
    estimation = sgd_estimate_update(lr, momentum, buffer, dampening, gradients, weight_decay, param, nesterov, maximize)
    print("===================================================================")
    print("estimation")
    print(estimation[0, 0, 0, 0])
    optimizer.step()
    print("===================================================================")
    print("step")
    print(param_group["params"][idx][0, 0, 0, 0])

    # test 2
    optimizer.zero_grad()
    scores = model(data_x)
    loss:torch.Tensor = criterion(scores, data_y)
    loss.backward()

    param_group = optimizer.param_groups[0]
    param:torch.Tensor = param_group["params"][idx]
    buffer = optimizer.state[param]["momentum_buffer"] if "momentum_buffer" in optimizer.state[param].keys() else None
    gradients = param.grad
    weight_decay=param_group['weight_decay']
    momentum=param_group['momentum']
    lr=param_group['lr']
    dampening=param_group['dampening']
    nesterov=param_group['nesterov']
    maximize=param_group['maximize']

    print("===================================================================")
    print("param")
    print(param[0, 0, 1, 0])
    estimation = sgd_estimate_update(lr, momentum, buffer, dampening, gradients, weight_decay, param, nesterov, maximize)
    print("===================================================================")
    print("estimation")
    print(estimation[0, 0, 1, 0])
    optimizer.step()
    print("===================================================================")
    print("step")
    print(param_group["params"][idx][0, 0, 1, 0])


def __test_adam_estimates__():
    import torchvision.models as models
    print("testing SGD estimation")
    seed_everything(1234)
    num_classes = 10
    batch_size = 128
    idx = 0

    import torchvision.models as models
    model:models.ResNet = models.resnet18()
    model.fc = nn.Linear(512, num_classes)
    
    # generate random input and outputs
    data_x = torch.randn(batch_size, 3, 224, 224)
    data_y = torch.randint(0, 10, (batch_size,)).to(torch.long)
    
    optimizer = torch.optim.Adam(model.parameters(), 0.1)
    criterion = nn.CrossEntropyLoss()
    
    # test 1: gradient accumulation in a batch
    optimizer.zero_grad()
    scores = model(data_x)
    loss:torch.Tensor = criterion(scores, data_y)
    loss.backward()

    param_group = optimizer.param_groups[0]
    param:torch.Tensor = param_group["params"][idx]
    gradients = param.grad
    param_state = optimizer.state[param]
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
    if len(param_state) == 0:
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        max_exp_avg_sq = torch.zeros_like(param)
        state_step = torch.tensor(0.)
    else:
        exp_avg = param_state["exp_avg"]
        exp_avg_sq = param_state["exp_avg_sq"]
        max_exp_avg_sq = param_state["max_exp_avg_sq"] if amsgrad else None
        state_step:torch.Tensor = param_state["step"]

    print("===================================================================")
    print("param")
    print(param[0, 0, 0, 0])
    estimation = adam_estimate_update(lr, gradients, weight_decay, param, maximize, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, amsgrad, beta1, beta2, eps, capturable, differentiable)
    print("===================================================================")
    print("estimation")
    print(estimation[0, 0, 0, 0])
    optimizer.step()
    print("===================================================================")
    print("step")
    print(param_group["params"][idx][0, 0, 0, 0])

    # test 2
    optimizer.zero_grad()
    scores = model(data_x)
    loss:torch.Tensor = criterion(scores, data_y)
    loss.backward()

    param_group = optimizer.param_groups[0]
    param:torch.Tensor = param_group["params"][idx]
    gradients = param.grad
    param_state = optimizer.state[param]
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
    if len(param_state) == 0:
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        max_exp_avg_sq = torch.zeros_like(param)
        state_step = torch.tensor(0.)
    else:
        exp_avg = param_state["exp_avg"]
        exp_avg_sq = param_state["exp_avg_sq"]
        max_exp_avg_sq = param_state["max_exp_avg_sq"] if amsgrad else None
        state_step:torch.Tensor = param_state["step"]

    print("===================================================================")
    print("param")
    print(param[0, 0, 1, 0])
    estimation = adam_estimate_update(lr, gradients, weight_decay, param, maximize, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step, amsgrad, beta1, beta2, eps, capturable, differentiable)
    print("===================================================================")
    print("estimation")
    print(estimation[0, 0, 1, 0])
    optimizer.step()
    print("===================================================================")
    print("step")
    print(param_group["params"][idx][0, 0, 1, 0])


if __name__ == "__main__":
    __test_adam_estimates__()