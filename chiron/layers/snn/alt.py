import torch
import torch.nn as nn


class SNNModel(nn.Module):
    def __init__(
        self, input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            timesteps: int = 100
    ):
        super(SNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.timesteps = timesteps
        self.mem = None
        self.spike = None
        self.hidden = None
        self.cell = None
        self.output = None


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        self.mem = out
        self.spike = torch.zeros_like(out)
        spikes = []
        for t in range(self.timesteps):
            self.spike = self.mem > 0.5
            self.mem = self.mem - self.spike
            spikes.append(self.spike)
            self.mem = self.mem + self.fc(self.spike)
            self.spike = self.mem > 0.5
            self.mem = self.mem - self.spike
            spikes.append(self.spike)


        spikes = torch.stack(spikes, dim=0)
        return spikes


