
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, encoder_layers = [3, 8, 64]):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(len(encoder_layers)-1):
            encoder.append( nn.Conv2d( encoder_layers[i], encoder_layers[i+1],
            kernel_size=3, stride= 2) )
            encoder.append( nn.Tanh() )
            encoder.append( nn.MaxPool2d(kernel_size=(3), 
            stride= (1), return_indices=True) )

        self.encoder = encoder
        self.Linear = nn.Linear(64*3*3,64)
 
    def forward(self, batch):
        indices = []
        out = batch
        sizes = []

        for i, layer in enumerate(self.encoder):
          if (i+1)%3 == 0:
            sizes.append(out.shape)
            out, index = self.encoder[i](out)
            indices.append(index)
            continue

          out = self.encoder[i](out)
        sizes.append(out.shape)
        out = out.reshape(batch.shape[0],-1)
        out = self.Linear(out)


        return out, indices,sizes

if __name__ == "__main__":
  a = Encoder()
  x,i = a(torch.zeros((1,3,28,28)))
  print(x.shape, i[-1].shape)
