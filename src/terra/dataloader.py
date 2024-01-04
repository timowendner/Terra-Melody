import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .utils import open_midi_file


class MIDIDataset(Dataset):
    def __init__(self, path: str, config: dict = None, desc: str = None) -> None:
        if config is None:
            raise AssertionError(
                'config file must be provided to create MIDI dataset'
            )
        files = glob.glob(os.path.join(path, '*.mid'))
        if 'debug size' in config:
            files = files[:config['debug size']]

        dataset = []
        for file in tqdm(files, desc=f'{desc}', ncols=80):
            sample = open_midi_file(file)
            sample = torch.Tensor(sample).float()
            dataset.append(sample)
        if len(dataset) == 0:
            raise AttributeError('Data path seems to be empty')

        self.device = config['device']
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.dataset[idx]
        sample = sample.to(self.device)
        return sample


def rnn_collate_fn(batch):
    sequence = pad_sequence(batch, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in batch])
    targets = sequence[:, 1:].contiguous()
    sequence = sequence[:, :-1].contiguous()
    return sequence, lengths-1, targets


def get_dataloaders(dataset: Dataset, config: dict, split: int) -> tuple[DataLoader, DataLoader]:
    batch_size = config['batch size']

    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=rnn_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False, collate_fn=rnn_collate_fn
    )

    return train_loader, test_loader
