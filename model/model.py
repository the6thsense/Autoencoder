import torch
from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder


class Model(nn.Module):
    def __init__(self, encoder_layers = [1,8,32, 64], decoder_layers = [64, 32,8, 1 ] ):
        super(Model,self).__init__()
        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)

    def forward(self, batch):
        latent_repr, indices,sizes = self.encoder(batch)
        output = self.decoder(latent_repr, indices,sizes)
        
        return output
    
    
if __name__ == "__main__":
  a = Model()
  x= a(torch.zeros((1,3,28,28)))
  print(x.shape)
