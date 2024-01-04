from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, config: dict):
        super(RNN, self).__init__()

        hidden_size = config['hidden size']
        input_size = 26
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(config['Dropout'])
        self.linear = nn.Linear(hidden_size, 26)
        self.octave = nn.Softmax(dim=2)
        self.key = nn.Softmax(dim=2)

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, enforce_sorted=False, batch_first=True
        )
        x, hidden = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        x = self.linear(x)
        x[5:15] = self.octave(x[5:15])
        x[15:] = self.key(x[15:])
        return x
