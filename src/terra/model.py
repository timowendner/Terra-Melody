from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, config: dict):
        super(RNN, self).__init__()

        bidirectional = config['bidirectional']
        hidden_size = config['hidden size']
        input_size = 5
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(config['Dropout'])
        hidden_size *= 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size, 5)

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, enforce_sorted=False, batch_first=True
        )
        x, hidden = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        x = self.linear(x)
        return x
