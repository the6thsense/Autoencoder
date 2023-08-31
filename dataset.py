from torchvision.datasets import  MNIST
from torchvision import transforms as t


train_transforms = t.Compose([ t.RandAugment(),
                        t.ToTensor(),
                        t.Normalize(mean = [0.1307], std = [0.3081]) ])

val_transforms = t.Compose([
                        t.ToTensor(),
                        t.Normalize(mean = [0.1307], std = [0.3081]) ])

train_data = MNIST(
    root = '/content/drive/MyDrive/autoencoder/Autoencoder/data/train',
    download= True, train = True, transform=train_transforms)
val_data = MNIST(
    root = '/content/drive/MyDrive/autoencoder/Autoencoder/data/test',
    download= True, train = False, transform=val_transforms)
