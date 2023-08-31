from torch.utils.data import DataLoader
from model.model import Model
import torch
from dataset import train_data, val_data
from torch import nn
from torch.optim import SGD
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

    trainloader = DataLoader(train_data,batch_size=args.batchsize, shuffle=True, )
    valloader = DataLoader(val_data,batch_size=args.batchsize, shuffle=True, )
    
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr = args.lr)
    train_losses = []

    from tqdm import tqdm
    for i in range(args.epochs):
        for l, batch in tqdm(enumerate(trainloader), total=  len(trainloader)):
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(batch, output)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if i%100:
                print('train mean loss: ------------>' ,np.mean(train_losses))
                train_losses = []

        val_losses = []
        with torch.no_grad():
            k = 0
            for batch in tqdm(valloader, total=  len(valloader)):
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(batch, output)
                if k<10 and random.randint(0,10) == 0:
                    Image.save(transforms.ToPILImage(output), f'{args.save_dir}/{k}.jpg')
                    k+=1
                if Path(args.save_dir).mkdir(exist_ok = True):
                    val_losses.append(loss.item())

        print('validation mean loss: ------------>' ,np.mean(val_losses))

if __name__= '__main__':
    parser = ArgumentParser(description="pass hyperparams from terminal")
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--batch-size', default=64)
    parser.add_argument('--save-dir', default="./predicted_samples")
    parser.add_argument('--lr', default=0.01)





