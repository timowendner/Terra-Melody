import yaml
import torch
import argparse
from torch import nn

from .dataloader import MIDIDataset, get_dataloaders
from .training_testing import train_network
from .model import RNN


def run(config_path: str, model: nn.Module = None) -> nn.Module:
    """Run a Neural Network model to predict the tempo of a MIDI file

    Args:
        config_path (str): path to the config file
        model (nn.Module, optional): already trained model. Defaults to None.

    Returns:
        nn.Module: the trained model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    model = RNN(config)
    model = model.to(device)

    data_path = config['train folder']
    dataset = MIDIDataset(data_path, config, desc='Loading Dataset')
    train_set, test_set = get_dataloaders(
        dataset, config, split=config['train split']
    )

    lr = config['learning rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer = train_network(
        model, optimizer, train_set, test_set, config
    )
    return model


def main():
    run(args.config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to the config file",
        type=str,
        default="config.yaml",
    )
    args = parser.parse_args()
    main()
