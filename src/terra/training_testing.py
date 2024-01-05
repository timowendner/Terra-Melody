import torch
import time
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer


from .dataloader import MIDIDataset
from .utils import CombinedLoss, get_midi


def train_network(
    model: nn.Module,
    optimizer: Optimizer,
    train_set: MIDIDataset,
    test_set: MIDIDataset,
    config: dict
) -> tuple[nn.Module, Optimizer]:
    """Train the network

    Args:
        model (nn.Module): the used model
        optimizer (Optimizer): the used optimizer
        train_set (MIDIDataset): the dataset to train on
        test_set (MIDIDataset): the dataset to test on
        config (dict): the config file

    Returns:
        tuple[nn.Module, Optimizer]: the trained model, the current optimizer
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    model.train()
    Loss = CombinedLoss()

    num_epoch = config['train epochs']
    for epoch in range(1, num_epoch+1):
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        desc = f'{time_now} Starting Epoch {epoch:>3}'
        for sample, length, target in tqdm(
            train_set, desc=f'{desc:<25}', ncols=80
        ):
            outputs = model(sample, length)
            loss = Loss(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = optimizer.param_groups[0]['lr']
        error = test_network(
            model, test_set, desc='      Testing Network'
        )
        print(f'      current error: {error:.4f}, lr: {lr}\n')
        get_midi(model, 10)
    return model, optimizer


def test_network(model: nn.Module, dataloader: DataLoader, desc: str = None) -> tuple[dict, float]:
    Loss = CombinedLoss()
    total_loss = 0
    model.eval()
    for sample, length, target in tqdm(
        dataloader, desc=f'{desc:<25}', ncols=80
    ):
        outputs = model(sample, length)
        loss = Loss(outputs, target)
        total_loss += loss.item() * len(sample)

    model.train()
    return total_loss / len(dataloader)
