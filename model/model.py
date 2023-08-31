import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder


class Model(nn.Module):
    def __init__(self, encoder_layers = [3,8, 64], decoder_layers = [64, 8, 3 ] ):
        super(Model,self).__init__()
        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)

    def forward(self, batch):
        latent_repr, indices,sizes = self.encoder(batch)
        print('decoder')
        output = self.decoder(latent_repr, indices,sizes)
        
        return output
    
    
if __name__ == "__main__":
  a = Model()
  x= a(torch.zeros((1,3,28,28)))
  print(x.shape)
