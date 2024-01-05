import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, config: dict):
        super(RNN, self).__init__()

        input_size = 6
        emb_size = config['embedding size']
        self.nonlinear = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        last = emb_size
        self.layers = nn.ModuleList([])
        for size in config['hidden sizes'] + [emb_size]:
            self.layers.append(
                nn.LSTM(last, size, batch_first=True)
            )
            last = size
        self.encoder = nn.Linear(input_size, emb_size)
        self.decoder = nn.Linear(emb_size, input_size)

    def forward(self, x, lengths):
        # apply encoder
        batch_size, seq_size, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.encoder(x)
        x = x.view(batch_size, seq_size, -1)

        x = pack_padded_sequence(
            x, lengths, enforce_sorted=False, batch_first=True
        )
        for layer in self.layers:
            x, _ = layer(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x.view(-1, x.shape[2])
        x = self.decoder(x)
        x = x.view(batch_size, seq_size, -1)

        # x[:, 4:14] = self.softmax(x[:, 4:14])
        # x[:, 14:] = self.softmax(x[:, 14:])
        # output = torch.zeros_like(x)
        # output[:, 15:] = x[:, 15:]
        return x
