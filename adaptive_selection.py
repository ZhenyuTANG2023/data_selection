import logging
import os
import time
import traceback
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from data_value.data_value import DataEvaluator, DataSelector
from utils.argument import Argument, parse_args
from utils.getters import (get_baseline_model, get_criterion, get_datasets,
                           get_lr_sched, get_optimizer)
from utils.logger import get_logger
from utils.training import (EarlyStopping, retrieve_classes, run_validation,
                            visualize_training)
from utils.utils import seed_everything


def training_loop(
    args:Argument,
    logger:logging.Logger,
    evaluator:DataEvaluator,
    selector:DataSelector,
    model:nn.Module,
    num_classes:int,
    train_dataset:Dataset,
    valid_dataset:Dataset,
    test_dataset:Dataset,
    optimizer:torch.optim.Optimizer,
    criterion:nn.modules.loss._Loss,
    params_to_update:List[torch.Tensor],
    params_to_update_names:List[str],
    save_folder:str,
    device:str,
    lrate_sched:torch.optim.lr_scheduler._LRScheduler,
    is_valid:bool,
):
    train_loss, valid_loss, train_epochs_loss, valid_epochs_loss = [], [], [], []
    lr_adjust = {}

    dv_save_folder = os.path.join(save_folder, "dv")
    model_save_folder = os.path.join(save_folder, "model")
    os.makedirs(dv_save_folder, exist_ok=True)
    os.makedirs(model_save_folder, exist_ok=True)

    train_dataloader_sampled = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10) if args.fullset else None
    test_dataloader = DataLoader(test_dataset, batch_size=(1 if args.dataset in ["imagenet", "indoor"] else args.batch_size), shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1) if valid_dataset is not None else None
    early_stopping = EarlyStopping(args.patience, logger) if is_valid else None

    train_indices = list(range(len(train_dataset)))
    logger.info(f"size of train indices: {len(train_indices)}")
    
    sel_start_epoch = -1
    if args.use_coreset:
        coreset_name = args.coreset
        logger.info(f"using {coreset_name} coreset")
        if coreset_name == "random":
            args.fullset = False
            evaluator._coreset = True
        else:
            train_indices = np.load(f"./data/coreset/{coreset_name}/{args.dataset}.npy").tolist()
            train_dataset = Subset(train_dataset, train_indices)
            train_dataloader_sampled = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10) if args.fullset else None
    
    logger.info(f"size of train indices: {len(train_indices)}")
    if valid_dataloader:
        logger.info(f"size of valid dataset: {len(valid_dataset)}")
    logger.info(f"size of test dataset: {len(test_dataset)}")

    iter_start_time = time.time()
    for epoch in range(args.num_epochs):
        if not args.fullset:
            if epoch > sel_start_epoch - 1:
                # evaluate and select data
                logger.info("getting observation data")
                feat_groups, feat_group_names, feat_group_nelements = evaluator.data_value(
                    train_indices,
                    model,
                    train_dataset,
                    num_classes,
                    optimizer,
                    criterion,
                    params_to_update,
                    epoch,
                    dv_save_folder
                )
                train_dataloader_sampled = selector.select_data(feat_groups, train_dataset, args.batch_size)
            else:
                train_dataloader_sampled = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
        if args.fullset and args.observe:
            logger.info("getting observation data")
            feat_groups, feat_group_names, feat_group_nelements = evaluator.data_value(
                train_indices,
                model,
                train_dataset,
                num_classes,
                optimizer,
                criterion,
                params_to_update,
                epoch,
                dv_save_folder
            )
            logger.info("got observation info")
        model.train()
        train_epoch_loss = []

        # train
        for idx, (x, y) in enumerate(train_dataloader_sampled):
            x:torch.Tensor = x.to(torch.float32).to(device)
            y:torch.Tensor = y.to(torch.float32 if args.multi_label else torch.long).to(device)
            scores = model(x)
            if args.multi_label:
                scores = torch.sigmoid(scores)
            loss:torch.Tensor = criterion(scores, y)
            loss.backward()

            if args.use_grad_clip:
                nn.utils.clip_grad_value_(params_to_update, args.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if len(train_dataloader_sampled) // 4 == 0 or idx % (len(train_dataloader_sampled) // 4) == 0:
                logger.info(f"epoch={epoch+1}/{args.num_epochs}, {idx}/{len(train_dataloader_sampled)} of train, loss={loss.item()}")
        
        selector.lrate_deadjust()
        if lrate_sched:
            lrate_sched.step()
        train_epochs_loss.append(np.average(train_epoch_loss))

        # valid
        if valid_dataset is not None:
            valid_epoch_loss = run_validation(logger, model, valid_dataloader, device, args.multi_label, criterion, "Validation", num_classes)
            valid_epochs_loss.append(np.sum(valid_epoch_loss))
            # early stopping
            if is_valid:
                early_stopping(valid_epochs_loss[-1], model, model_save_folder)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            logger.info(f"Updating learning rate to {lr}")
        
        # estimate time remaining since tqdm is not accessible in the cluster servers
        if epoch < args.num_epochs - 1:
            now_time = time.time()
            print(f"time: {now_time - iter_start_time:.3f} s passed\n{(now_time - iter_start_time) / (epoch + 1) * (args.num_epochs - epoch - 1):.3f} s left")
        torch.save(model.state_dict(), os.path.join(model_save_folder, f"epoch_{epoch}.pt"))
    
    if valid_dataset is not None and is_valid:
        model.load_state_dict(early_stopping.saved_params)

    # test
    run_validation(logger, model, test_dataloader, device, args.multi_label, criterion, "Test", num_classes)

    return train_loss, valid_loss, train_epochs_loss, valid_epochs_loss


def train(args:Argument):
    logger, timestamp = get_logger(logger_name=args.info)
    logger.info(args)

    start_time = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
        logger.info(f"device: {device}")

        # for replay
        seed_everything(args.rnd_seed)

        # so long as the seed keeps the same, the split will be the same
        is_valid, train_dataset, test_dataset, valid_dataset = get_datasets(args.dataset, args.valid_ratio, args.data_folder)
        num_classes, classes = retrieve_classes(test_dataset)
        logger.info("number of classes and classes: " + str(num_classes) + ", " + str(classes))

        model = get_baseline_model(args.model, num_classes, args.pretrained)
        model.to(device)
        params_to_update:List[torch.Tensor] = []
        params_to_update_names:List[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                params_to_update_names.append(name)
                logger.info(f"\t{name}")
        

        optimizer = get_optimizer(args.optimizer, params_to_update, args.lrate, args.weight_decay, args.momentum)
        criterion = get_criterion(args.criterion).to(device)
        lrate_sched = get_lr_sched(args.lr_sched, optimizer, args.num_epochs)
        
        valuator = DataEvaluator(
            eval_method=args.data_val_method, 
            data_val_feat=args.data_val_dims,
            multi_label=args.multi_label, 
            valuation_batch_size=args.data_val_batch_size, 
            device=device, 
            params_to_update_names=params_to_update_names,
            run_with_slurm=args.run_with_slurm,
            logger=logger
        )
        
        selector = DataSelector(
            args.selection_strategy,
            args.sample_size,
            args.lr_adjust,
            optimizer,
            logger
        )

        train_loss, valid_loss, train_epochs_loss, valid_epochs_loss = training_loop(args, logger, valuator, selector, model, num_classes, train_dataset, valid_dataset, 
            test_dataset, optimizer, criterion, params_to_update, params_to_update_names, os.path.join("./data/verf", args.dataset, args.model), device, lrate_sched, is_valid)

    except Exception as e:
        logger.error("Error occurs! Error message:")
        logger.error(e)
        logger.info("Traceback information:")
        logger.info(traceback.print_exc())
    
    finally:
        if args.vis and len(train_epochs_loss):
            visualize_training(
                logger, 
                train_loss, 
                train_epochs_loss, 
                valid_epochs_loss if valid_dataset is not None else None,
                timestamp
            )

        logger.info(f"training elasped in {time.time() - start_time:.3f} s")

if __name__ == "__main__":
    ARGS = parse_args(default_config_file="./config/default.yaml")
    train(ARGS)
