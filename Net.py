import torch
import torch.nn as nn
import math
from torchdiffeq import odeint

class SCM(nn.Module):
    def __init__(self, num_layers, rnn_hidden_size, encoder_input_size, encoder_hidden_size, depth=300):
        super(MLP_BiGRU, self).__init__()
        self.depth = depth
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.embedding_lat = nn.Embedding(64, 16)
        self.embedding_lon = nn.Embedding(360, 16)
        self.embedding_sst = nn.Embedding(64, 16)
        self.embedding_date = nn.Embedding(2048, 16)

        self.mlp = nn.Sequential(
            nn.Linear(encoder_input_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, depth),
        )

        self.bi_gru = nn.GRU(input_size=1, hidden_size=rnn_hidden_size,
                                  num_layers=num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(rnn_hidden_size * 2 * depth, depth)

        self.flatten = nn.Flatten(1, -1)

    def forward(self, x):
        x0 = torch.cat((self.embedding_lat(torch.trunc(x[:, 0]).long()), torch.frac(x[:, 0]).unsqueeze(1)), 1)
        x1 = torch.cat((self.embedding_lon(torch.trunc(x[:, 1]).long()), torch.frac(x[:, 1]).unsqueeze(1)), 1)
        x2 = self.embedding_date(x[:, 2].long())
        x3 = torch.cat((self.embedding_sst(torch.trunc(x[:, 3]).long()), torch.frac(x[:, 3]).unsqueeze(1)), 1)
        x = torch.cat((x0, x1, x2, x3, x[:, -1].unsqueeze(1)), 1)
        x = self.mlp(x.float())
        output, h = self.bi_gru(x.unsqueeze(-1))
        output = self.flatten(output)
        output = self.fc(output)

        return output