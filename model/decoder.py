
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, decoder_layers = [64, 8, 3 ]):
        super().__init__()
        self.decoder_layers = decoder_layers
        decoder = []
        k = 0
        for i in range(len(decoder_layers)-1):
            decoder.append( nn.MaxUnpool2d(kernel_size = 5, stride = 1))
            decoder.append( nn.ConvTranspose2d( decoder_layers[i], decoder_layers[i+1],
            kernel_size = 5, stride = 1)) 
            decoder.append( nn.Tanh() )

        self.decoder = decoder
        self.Linear = nn.Linear(decoder_layers[0], 4*4*decoder_layers[0])
        self.upsample = nn.Upsample(size = (28,28), mode='bilinear')
    def forward(self, batch, indices,sizes): 
        out = self.Linear(batch)
        out = out.reshape(batch.shape[0],self.decoder_layers[0], 4,4)
        k = len(indices)-1
        for i,layer in enumerate(self.decoder):
          if i%3 == 0:
            out = self.decoder[i](out, indices = indices[k], output_size = sizes[k])
            k -= 1
            continue

          out = self.decoder[i](out)
        out = self.upsample(out)

        return out

if __name__ == "__main__":
  a = Decoder()(torch.zeros((1,64)))
