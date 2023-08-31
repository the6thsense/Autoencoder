from torch.utils.data import DataLoader
from model.model import Model
import torch
from dataset import train_data, val_data
from torch import nn
from torch.optim import Adam
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from torchvision import transforms
import cv2
import random
from PIL import Image


def train(args):
    
    model = Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # model.load_state_dict(torch.load("/content/drive/MyDrive/autoencoder/Autoencoder/model.pth"))
    trainloader = DataLoader(train_data,batch_size=args.batchsize, shuffle=True, )
    valloader = DataLoader(val_data,batch_size=args.batchsize, shuffle=True, )
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = args.lr)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, gamma = 0.1, step_size=5)
    train_losses = []
    min_val_loss = 1000

    from tqdm import tqdm
    for i in range(args.epochs):
        print(f"epoch: {i}")
        for l, batch in tqdm(enumerate(trainloader), total=  len(trainloader)):
            batch = batch[0]
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(batch, output)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if l%10== 0:
                print('train mean loss: ------------>' ,np.mean(train_losses))
                train_losses = []

        print('train mean loss: ------------>' ,np.mean(train_losses))
        train_losses = []
        lr_schedule.step()
        print('lr----------------------------->',lr_schedule.get_last_lr())
        val_losses = []
        with torch.no_grad():
            k = 0
            for batch in tqdm(valloader, total=  len(valloader)):
                batch = batch[0].to(device)
                output = model(batch)
                loss = criterion(batch, output)
                val_losses.append(loss.item())
                for img in output:
                  if k<10 and random.randint(0,10) == 0:
                      Path(args.save_dir).mkdir(exist_ok = True)
                      transforms.ToPILImage()(img).save( f'{args.save_dir}/{k}.jpg')
                      k+=1
        val_loss = np.mean(val_losses)
        print('validation mean loss: ------------>' ,val_loss)
        if min_val_loss>val_loss:
          print(f'saving checkpoint: {np.mean(val_losses)} {min_val_loss}')
          torch.save(model.state_dict(), "./model.pth")
          min_val_loss = val_loss


if __name__== '__main__':
    parser = ArgumentParser(description="pass hyperparams from terminal")
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--batchsize', default=512)
    parser.add_argument('--save-dir', default="./predicted_samples")
    parser.add_argument('--lr', default=0.01)

    args = parser.parse_args()
    train(args)



