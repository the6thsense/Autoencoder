
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, encoder_layers = [3,4, 8, 16, 32, 64]):
        encoder = []
        for i in range(len(encoder_layers)):
            encoder.append( nn.Conv2d( encoder_layers[i], encoder_layers[i+1], kernel_size=(), stride= ()) )
            encoder.append( nn.Tanh() )
            encoder.append( nn.MaxPool2d(kernel_size=(), stride= ()) )

        self.encoder = nn.Sequential(encoder)

    def forward(self, batch):
        return self.encoder(batch)

