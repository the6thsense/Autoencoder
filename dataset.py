from torchvision.datasets import  MNIST
from torchvision import transforms as t


transforms = t.Compose([ t.RandAugment(),
                        t.ToTensor(),
                        t.Normalize(mean = [0.1307], std = [0.3081]) ])

train_data = MNIST(
    root = '/content/drive/MyDrive/autoencoder/Autoencoder/data/test',
    download= True, train = True, transform=transforms)
val_data = MNIST(
    root = '/content/drive/MyDrive/autoencoder/Autoencoder/data/test',
    download= True, train = True
    )
