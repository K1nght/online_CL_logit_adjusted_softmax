import math
import sys
from argparse import Namespace
from typing import Tuple
from datetime import datetime

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar
from utils.metrics import forgetting, balanced_error

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    mask = torch.ones(dataset.N_CLASSES)
    mask[dataset.CLASSES_PER_TASK[k]] = 0
    outputs[:, (mask == 1)] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    total_preds, total_targets = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        # print(k)
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                
                _, pred = torch.max(outputs.data, 1)
                total_preds.append(pred)
                total_targets.append(labels)
                # print(outputs[:5], pred[:5])
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    total_preds = torch.concat(total_preds, dim=0)
    total_targets = torch.concat(total_targets, dim=0)

    model.net.train(status)
    print(f"class-il acc {accs}")
    if dataset.SETTING == 'class-il':
        print(f"task-il acc {accs_mask_classes}")
    if dataset.NAME == 'seq-inat':
        bal_err = balanced_error(total_preds, total_targets)
        print(f"balanced acc {bal_err * 100}")
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()
        if not hasattr(args, 'buffer_size'):
            wandb.run.name = "_".join([
                str(item) 
                for item in [args.dataset, args.model, -1, datetime.now().strftime("%Y%m%d-%H%M%S")]])
        else:
            wandb.run.name = "_".join([
                str(item) 
                for item in [args.dataset, args.model, args.buffer_size, datetime.now().strftime("%Y%m%d-%H%M%S")]])

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if 'knn' not in model.NAME and model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    class_mean_accs = []
    task_mean_accs = []
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if type(not_aug_inputs) == torch.Tensor:
                        not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if type(not_aug_inputs) == torch.Tensor:
                        not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)

        class_mean_accs.append(mean_acc[0])
        task_mean_accs.append(mean_acc[1])
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)

    if dataset.SETTING == 'class-il':
        print(f"forget: {forgetting(results)}, masked forget: {forgetting(results_mask_classes)}")
    elif dataset.SETTING == 'blurry-class-il':
        print(f"average acc: {np.mean(np.array(class_mean_accs))}")

    if not args.disable_log:
        logger.add_forgetting(results, results_mask_classes)
        if not args.ignore_other_metrics:
            logger.add_bwt(results, results_mask_classes)
            if 'knn' not in model.NAME and model.NAME != 'icarl' and model.NAME != 'pnn':
                logger.add_fwt(results, random_results_class,
                        results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
