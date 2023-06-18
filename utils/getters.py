import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, random_split
from torchvision import datasets

from .cls_dataset import (CRC100K, ISIC2019, ISIC2020, MNIST_REP, ChestXRay,
                          ImageNet, KvasirCapsule, MITIndoor)
from .training import set_bn_eval, set_bn_train


def get_datasets(dataset:str, valid_ratio:float, data_folder:str):
    """
        Get the corresponding baseline datasets given the dataset name.

        Params:
        ---------
        dataset:        the name of the dataset
        valid_ratio:    the ratio of the valid set, if no valid set should be used, set to 0.
        
        Returns:
        ----------
        is_valid:       whether a valid dataset is split from the training set
        train_dataset:  the training dataset
        test_dataset:   the testing dataset
        valid_dataset:  the validation dataset
    """
    valid_dataset = None

    if dataset == "svhn":
        train_dataset = datasets.SVHN(os.path.join(data_folder, dataset), download=False, split="train", transform=transforms.ToTensor())
        test_dataset = datasets.SVHN(os.path.join(data_folder, dataset), download=False, split="test", transform=transforms.ToTensor())
    elif dataset == "cifar10":
        mean = np.array([125.3, 123.0, 113.9]) / 255.0
        std = np.array([63.0, 62.1, 66.7]) / 255.0
        # for fair comparison, apply the image preprocessing step as in
        #   He et al. Deep Residual Learning for Image Recognition.
        #       https://arxiv.org/pdf/1512.03385.pdf
        transform = transforms.Compose([
            transforms.RandomCrop(32, 4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        norm_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.CIFAR10(os.path.join(data_folder, dataset), download=False, train=True, transform=transform)
        test_dataset = datasets.CIFAR10(os.path.join(data_folder, dataset), download=False, train=False, transform=norm_transform)
    elif dataset == "cifar100":
        mean = np.array([125.3, 123.0, 113.9]) / 255.0
        std = np.array([63.0, 62.1, 66.7]) / 255.0
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode="reflect"),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        norm_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.CIFAR100(os.path.join(data_folder, dataset), download=False, train=True, transform=transform)
        test_dataset = datasets.CIFAR100(os.path.join(data_folder, dataset), download=False, train=False, transform=norm_transform)
    elif dataset == "mnist":
        train_dataset = MNIST_REP(os.path.join(data_folder, dataset), download=False, train=True, transform=transforms.ToTensor())
        test_dataset = MNIST_REP(os.path.join(data_folder, dataset), download=False, train=False, transform=transforms.ToTensor())
    elif dataset == "chestxray14":
        norm_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ChestXRay(train=True, transform=norm_transform, petrel_oss=False)
        test_dataset = ChestXRay(train=False, transform=norm_transform, petrel_oss=False)
    elif dataset == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomChoice([transforms.Resize(256), transforms.Resize(480)]),
            transforms.RandomCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageNet(os.path.join(data_folder, dataset), split="train", transform=train_transform)
        test_dataset = ImageNet(os.path.join(data_folder, dataset), split="val", transform=valid_transform)
    elif dataset == "indoor":
        train_transform = transforms.Compose([
            transforms.RandomChoice([transforms.Resize(256), transforms.Resize(480)]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = MITIndoor(os.path.join(data_folder, dataset), split="train", transform=train_transform)
        test_dataset = MITIndoor(os.path.join(data_folder, dataset), split="val", transform=valid_transform)
    elif dataset == "kvasir_capsule":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Kvasir Capsule does not provide an official train-val split
        # here we split it into train:test=8:2
        full_dataset = KvasirCapsule(os.path.join(data_folder, dataset), transform=valid_transform)
        full_size = len(full_dataset)
        train_size = int(0.9 * full_size)
        test_size = full_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        # train_dataset.transform = train_transform
        setattr(train_dataset, "classes", full_dataset.classes)
        setattr(test_dataset, "classes", full_dataset.classes)
    elif dataset == "crc100k":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # CRC100K does not provide an official train-val split
        # here we split it into train:test=8:2
        full_dataset = CRC100K(os.path.join(data_folder, "NCT-CRC-HE-100K"), transform=valid_transform)
        full_size = len(full_dataset)
        train_size = int(0.9 * full_size)
        test_size = full_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        setattr(train_dataset, "classes", full_dataset.classes)
        setattr(test_dataset, "classes", full_dataset.classes)
    elif dataset == "isic2020":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # transforms.RandomRotation(30),
            # # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # CRC100K does not provide an official train-val split
        # here we split it into train:test=8:2
        full_dataset = ISIC2020(os.path.join(data_folder, "isic_2020"), transform=valid_transform)
        full_size = len(full_dataset)
        train_size = int(0.9 * full_size)
        test_size = full_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        # train_dataset.transform = train_transform
        setattr(train_dataset, "classes", full_dataset.classes)
        setattr(test_dataset, "classes", full_dataset.classes)
    elif dataset == "isic2019":
        # https://github.com/pranavsinghps1/CASS/blob/master/CASS.ipynb
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomRotation(30),
            # # transforms.RandomResizedCrop(224),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2),
                 transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2),
                 transforms.RandomAffine(degrees=10),], p=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # CRC100K does not provide an official train-val split
        # here we split it into train:test=8:2
        full_dataset_train = ISIC2019(os.path.join(data_folder, "isic_2019"), transform=train_transform)
        full_dataset_test = ISIC2019(os.path.join(data_folder, "isic_2019"), transform=valid_transform)
        full_size = len(full_dataset_train)
        indices = list(range(full_size))
        train_indices, test_indices = train_test_split(indices, train_size=0.8)
        train_dataset = Subset(full_dataset_train, train_indices)
        test_dataset = Subset(full_dataset_test, test_indices)
        setattr(train_dataset, "classes", full_dataset_train.classes)
        setattr(test_dataset, "classes", full_dataset_train.classes)
    else:
        raise NotImplementedError
        
    is_valid = valid_ratio > 0. and valid_ratio < 1.
    if is_valid and valid_dataset is None:
        valid_size = int(len(train_dataset) * valid_ratio)
        train_size = len(train_dataset) - valid_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    return is_valid, train_dataset, test_dataset, valid_dataset


def init_layer(m:nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


def set_parameter_requires_grad(model:nn.Module, feature_extracting:bool=True, left_out_weights:List[str]=[]):
    if feature_extracting:
        for param_name, param in model.named_parameters(): 
            if param_name not in left_out_weights: continue
            param.requires_grad = False


def get_baseline_model(model_name:str, num_classes:int, pretrained:bool):
    if model_name == "resnet18":
        if pretrained:
            model:models.ResNet = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model:models.ResNet = models.resnet18(num_classes=num_classes)
    elif model_name == "resnet50":
        if pretrained:
            model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet50(num_classes=num_classes)
    elif model_name == "resnet101":
        if pretrained:
            model = models.resnet101(weights=models.resnet.ResNet101_Weights.IMAGENET1K_V2, num_classes=num_classes)
        else:
            model = models.resnet101(num_classes=num_classes)
    elif model_name == "efficientnet_v2_m":
        if pretrained:
            model = models.efficientnet_v2_m(weights=models.efficientnet.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_v2_m()
        model.classifier._modules["1"] = nn.Linear(model.classifier._modules["1"].in_features, num_classes)
    elif model_name == "efficientnet_v2_l":
        if pretrained:
            model = models.efficientnet_v2_l(weights=models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_v2_l()
        model.classifier._modules["1"] = nn.Linear(model.classifier._modules["1"].in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        if pretrained:
            model = models.efficientnet_v2_s(weights=models.efficientnet.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_v2_s()
        model.classifier._modules["1"] = nn.Linear(model.classifier._modules["1"].in_features, num_classes)
    elif model_name == "vit_b_16":
        if pretrained:
            model:models.VisionTransformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            model:model.VisionTransformer = models.vit_b_16()
        model.heads._modules["head"] = nn.Linear(model.heads._modules["head"].in_features, num_classes)
    elif model_name == "vit_b_32":
        if pretrained:
            model:models.VisionTransformer = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        else:
            model:model.VisionTransformer = models.vit_b_32()
        model.heads._modules["head"] = nn.Linear(model.heads._modules["head"].in_features, num_classes)
    else:
        raise NotImplementedError

    return model


def get_lr_sched(lr_sched:str, optimizer:torch.optim.Optimizer, num_epochs:int) -> torch.optim.lr_scheduler._LRScheduler:
    if lr_sched == "none":
        return None
    elif lr_sched == "multistep_lr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(num_epochs // 4) * 2, (num_epochs // 4) * 3])
    elif lr_sched == "cosine_lr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    raise NotImplementedError


def get_optimizer(optimizer:str, parameters, lrate:float, weight_decay:float, momentum:float) -> torch.optim.Optimizer:
    if optimizer == "adam":
        return torch.optim.Adam(parameters, lrate, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return torch.optim.AdamW(parameters, lrate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return torch.optim.SGD(parameters, lrate, momentum, weight_decay=weight_decay)
    raise NotImplementedError


def get_criterion(criterion:str) -> nn.modules.loss._Loss:
    if criterion == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif criterion == "mse":
        return nn.MSELoss()
    elif criterion == "bce":
        return nn.BCELoss()
    raise NotImplementedError