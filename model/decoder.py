
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, decoder_layers = [64, 32, 8, 3 ]):
        decoder = []
        for i in range(len(decoder_layers)):

            decoder.append( nn.ConvTranspose2d( decoder_layers[i], decoder_layers[i+1])) )
            decoder.append( nn.MaxUnpool2d(kernel_size= ()))
            decoder.append( nn.Tanh() )

        self.decoder = nn.Sequential(decoder)

    def forward(self, batch):
        return self.decoder(batch)

