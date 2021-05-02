import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from .helper import ProgressBar


def _is_binary(outputs):
    return len(outputs.shape) == 1 or outputs.shape[1] == 1


def _do_criterion(criterion, outputs, labels):
    if _is_binary(outputs):
        outputs = outputs.view(-1)
        return criterion(outputs, labels * 1.0)
    return criterion(outputs, labels)


def _do_prediction(outputs):
    if _is_binary(outputs):
        outputs = outputs.view(-1)
        return (outputs > 0.5) * 1
    return outputs.max(dim=1)[1]


def validate_model(model, validloader, criterion, device='cuda'):
    model.to(device)
    model.eval()

    test_loss = 0
    accuracy = 0
    total = 0

    with torch.no_grad():
        for batch in validloader:
            batch = [
                data.to(device) if torch.is_tensor(data) else data
                for data in batch]
            inputs = batch[:-1]
            labels = batch[-1]

            outputs = model.forward(*inputs)
            test_loss += _do_criterion(criterion, outputs, labels).item()

            equality = (labels.data == _do_prediction(outputs))
            accuracy += equality.type(torch.FloatTensor).sum()
            total += len(labels)

    return float(test_loss/len(validloader)), float(accuracy/total)


def save_model(model: nn.Module, checkpoint_path: str):
    torch.save(model.state_dict(), checkpoint_path)


def load_model(model: nn.Module, checkpoint_path: str) -> nn.Module:
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def _running_average(data, last_n):
    window = data[-last_n:]
    return sum(window) / len(window)


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    epochs: int,
    device: str = 'cuda',
    valid_every: Union[int, str] = 'epoch'
):
    session = datetime.now().strftime('%Y-%m-%d_%H-%M')
    checkpoints_dir = os.path.join('checkpoints', session)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model.to(device)

    if valid_every == 'epoch':
        valid_every = len(trainloader)
    else:
        valid_every = int(valid_every)

    steps = 0
    train_loss_history = []
    train_accuracy_history = []
    valid_loss_history = []
    valid_accuracy_history = []

    def dumpy_history():
        return {
            'epoch_size': len(trainloader),
            'valid_every': valid_every,
            'total_steps': steps,
            'train_loss_history': train_loss_history,
            'train_accuracy_history': train_accuracy_history,
            'valid_loss_history': valid_loss_history,
            'valid_accuracy_history': valid_accuracy_history,
        }

    for e in range(epochs):
        print(f'Epoch: {e+1}/{epochs} @ {datetime.now()}')
        model.train()
        validated = False

        progressbar = ProgressBar()
        for batch in progressbar(trainloader):
            steps += 1

            # Prepare data
            batch = [
                data.to(device) if torch.is_tensor(data) else data
                for data in batch]
            inputs = batch[:-1]
            labels = batch[-1]

            # Forward
            optimizer.zero_grad()
            outputs = model.forward(*inputs)

            # Backward
            loss = _do_criterion(criterion, outputs, labels)
            loss.backward()
            optimizer.step()

            # Record loss and accuracy
            train_loss_history.append(loss.item())
            equality = (labels.data == _do_prediction(outputs))
            accuracy = equality.type(torch.FloatTensor).mean()
            train_accuracy_history.append(float(accuracy))

            if steps % valid_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    valid_loss, valid_accuracy = validate_model(
                        model, validloader, criterion, device)

                valid_loss_history.append(valid_loss)
                valid_accuracy_history.append(valid_accuracy)

                # Dump history
                history = dumpy_history()
                with open(os.path.join(checkpoints_dir, 'history.json'), mode='w') as fp:
                    json.dump(history, fp)
                fig = plt.figure(figsize=(8, 8))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                _plot_history(history, ax1, ax2)
                ax1.legend()
                ax2.legend()
                fig.savefig(os.path.join(checkpoints_dir, 'history.png'))
                plt.close(fig)

                # Make sure training is back on
                model.train()
                validated = True

            # Update progress bar
            avg_loss = _running_average(train_loss_history, valid_every)
            avg_accuracy = _running_average(train_accuracy_history, valid_every)
            content = [
                'loss: {:.3f}'.format(avg_loss),
                'accuracy: {:.3f}'.format(avg_accuracy)
            ]
            if validated:
                content.extend([
                    'val_loss: {:.3f}'.format(valid_loss_history[-1]),
                    'val_accuracy: {:.3f}'.format(valid_accuracy_history[-1])
                ])
            progressbar.set_content(' - '.join(content))

            # Save checkpoint
            if progressbar.last_item:
                checkpoint_path = os.path.join(checkpoints_dir, f'ckpt_{e:02d}.pth')
                model_states = model.state_dict()
                torch.save(model_states, checkpoint_path)
                progressbar.content += f' - saved to {checkpoint_path}'

    return dumpy_history()


def _plot_history(history, ax_loss, ax_accuracy, label_suffix='', index=-1):
    epoch_size = history['epoch_size']
    valid_every = history['valid_every']
    train_loss_history = history['train_loss_history']
    valid_loss_history = history['valid_loss_history']
    train_accuracy_history = history['train_accuracy_history']
    valid_accuracy_history = history['valid_accuracy_history']

    train_steps = np.array(range(valid_every, len(train_loss_history) + 1)) / epoch_size
    valid_steps = (np.array(range(len(valid_loss_history))) + 1) * valid_every / epoch_size

    train_loss_history = np.convolve(
        train_loss_history, np.ones(valid_every)/valid_every, mode='valid'
    )
    train_accuracy_history = np.convolve(
        train_accuracy_history, np.ones(valid_every)/valid_every, mode='valid'
    )

    train_style, valid_style = {}, {}
    if index >= 0:
        train_style = {
            'color': list(mcolors.TABLEAU_COLORS)[index],
        }
        valid_style = {
            'color': list(mcolors.TABLEAU_COLORS)[index],
            'ls': '--'
        }
    ax_loss.plot(train_steps, train_loss_history,
                 label='training loss'+label_suffix, **train_style)
    ax_loss.plot(valid_steps, valid_loss_history,
                 label='validation loss'+label_suffix, **valid_style)
    ax_accuracy.plot(train_steps, train_accuracy_history,
                     label='training accuracy'+label_suffix, **train_style)
    ax_accuracy.plot(valid_steps, valid_accuracy_history,
                     label='validation accuracy'+label_suffix, **valid_style)


def plot_history(history: Dict[str, Any]) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    _plot_history(history, ax1, ax2)
    ax1.legend()
    ax2.legend()
    plt.show()


def plot_history_from_checkpoints(checkpoints: List[str]) -> None:
    history = None
    for ckpt in checkpoints:
        with open(os.path.join(ckpt, 'history.json'), 'r') as fp:
            h = json.load(fp)
        if history is None:
            history = h
        else:
            history['total_steps'] = h['total_steps']
            for key in (
                'train_loss_history',
                'train_accuracy_history',
                'valid_loss_history',
                'valid_accuracy_history'
            ):
                history[key].extend(h[key])
    plot_history(history)


def plot_histories(histories: Dict[str, Dict[str, Any]], loss_ylim=(0.0, 0.8)):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    if loss_ylim:
        ax1.set_ylim(loss_ylim)
    ax2 = fig.add_subplot(212)
    for i, (annotation, history) in enumerate(histories.items()):
        _plot_history(history, ax1, ax2, f' ({annotation})', i)
    ax1.legend()
    ax2.legend()
    plt.show()
