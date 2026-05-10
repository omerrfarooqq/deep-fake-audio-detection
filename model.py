import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), pool=(2,2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool)
        )

    def forward(self,x):
        return self.net(x)

class CNNLSTM(nn.Module):
    def __init__(self, n_mfcc=40, lstm_hidden=128, lstm_layers=1, bidirectional=False):
        super().__init__()
        # Input shape: (batch, n_mfcc, time)
        self.conv1 = CNNBlock(1, 16)
        self.conv2 = CNNBlock(16, 32)
        self.conv3 = CNNBlock(32, 64)
        # After 3x (2,2) pooling, freq dimension is reduced by 8
        self.freq_bins = max(1, n_mfcc // 8)
        self.lstm = nn.LSTM(input_size=64 * self.freq_bins, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, n_mfcc, time)
        x = x.unsqueeze(1)  # -> (batch, 1, n_mfcc, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x shape now (batch, channels, freq, time)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, freq)
        x = x.view(b, t, c * f)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.classifier(last).squeeze(1)
        return logits
