# Model from: https://arxiv.org/pdf/1511.06434.pdf
# Generator architecture: https://www.researchgate.net/profile/Hamed-Alqahtani/publication/338050169/figure/fig10/AS:849390331756554@1579521844631/depicts-the-DCGAN-generator-for-LSUN-sample-scene-modeling-The-DCGAN-models-performance.ppm
# Code based off https://github.com/bvshyam/facegeneration_gan_sagemaker/blob/master/train/model.py

import argparse
import json
import os
import pickle
import sys
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pickle as pkl
from torchvision import transforms
from torchvision import datasets

try:
    from train.model import DataDiscriminator, DataGenerator
except:
    from model import DataDiscriminator, DataGenerator


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    print("Model directory:", model_dir)

    # Determine the device and construct the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = DataGenerator(z_size=100, conv_dim=64)  # TODO change to 128/64?

    model_info = {}
    model_info_path = os.path.join(model_dir, 'generator_model.pt')

    with open(model_info_path, 'rb') as f:
        G.load_state_dict(torch.load(f))

    G.to(device).eval()

    print("Done loading model.")
    return G


def real_loss(D_out, train_on_gpu, smooth=False):

    batch_size = D_out.size(0)

    # Label smoothing
    if smooth:
        # Smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size)  # Real labels = 1
    # Move labels to GPU if available
    if train_on_gpu:
        labels = labels.cuda()
    # Binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()

    # Calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # Fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # Calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # Assume x is scaled to (0, 1)
    # Scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min
    return x


def get_dataloader(batch_size, image_size, data_dir):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """

    classname = m.__class__.__name__

    if classname == 'Conv2d':

        w = torch.empty(m.weight.data.shape)
        m.weight.data = nn.init.kaiming_uniform_(w)

    if classname == 'Linear':
        w = torch.empty(m.weight.data.shape)
        m.weight.data = nn.init.kaiming_uniform_(w)


def build_network(d_conv_dim, g_conv_dim, z_size):
    # Define discriminator and generator
    D = DataDiscriminator(d_conv_dim)
    G = DataGenerator(z_size=z_size, conv_dim=g_conv_dim)

    # Initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)

    return D, G


def real_loss(D_out, train_on_gpu, smooth=False):

    batch_size = D_out.size(0)

    # Label smoothing
    if smooth:
        # Smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size)  # Real labels = 1
    # Move labels to GPU if available
    if train_on_gpu:
        labels = labels.cuda()
    # Binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()

    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def train(D, G, z_size, train_loader, epochs, d_optimizer, g_optimizer, train_on_gpu):
    '''Training method called by PyTorch training script'''

    print_every = 50
    losses = []

    if train_on_gpu:
        D.cuda()
        G.cuda()

    # Fixed data for sampling, used in each epoch to generate images and can be compared across epochs
    samples = []
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    print('Number epochs:', epochs)

    # Training
    for epoch in range(epochs):

        for batch_i, (real_images, _) in enumerate(train_loader):

            batch_size = real_images.size()[0]

            real_images = scale(real_images)

            # -------- TRAIN DISCRIMINATOR --------

            D.train()
            G.train()

            d_optimizer.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)

            d_real_loss = real_loss(D_real, train_on_gpu)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # Move x to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_data = G(z)

            # Compute the discriminator losses on fake images
            D_fake = D(fake_data)
            d_fake_loss = fake_loss(D_fake, train_on_gpu)

            # Add up loss, backpropogation
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # -------- TRAIN GENERATOR --------

            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute discriminator losses on fake images (using flipped labels)
            D_fake = D(fake_images)
            # use real loss to flip labels
            g_loss = real_loss(D_fake, train_on_gpu)

            # Backpropogation
            g_loss.backward()
            g_optimizer.step()

            # Append/print generator/discriminator losses
            if batch_i % print_every == 0:
                losses.append((d_loss.item(), g_loss.item()))
                print('Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, epochs, batch_i+1, len(train_loader), d_loss.item(), g_loss.item()))

        # Generate and save sample, fake images after each epoch
        G.eval()
        if train_on_gpu:
            fixed_z = fixed_z.cuda()

        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()

    print(samples_z)

    with open('generator_model.pt', 'wb') as f:
        torch.save(G.state_dict(), f)

    print('Saved model at', f)

    # Save final epoch training generator sample images
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    print('Saved sample generated images at', f)

    return G


if __name__ == '__main__':
    # Model parameters and training parameters sent as arguments when script executed
    # Argument parser to access parameters

    # Mostly from https://arxiv.org/pdf/1511.06434.pdf !

    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--z_size', type=int, default=100, metavar='N',
                        help='input z-size for training (default: 100)')

    # Model parameters
    parser.add_argument('--conv_dim', type=int, default=64, metavar='N',
                        help='size of the convolution dim (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='N',
                        help='Learning rate default 0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='N',
                        help='beta1 default value 0.5')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='N',
                        help='beta2 default value 0.999')
    # parser.add_argument('--img_size', type=int, default=32, metavar='N',
    #                     help='Image size default value 32')
    parser.add_argument('--img_size', type=int, default=64, metavar='N',
                        help='Image size default value 64')  # TODO orig 32

    # SageMaker parameters
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])

    parser.add_argument('output_path', metavar='output_path', type=str)

    args = parser.parse_args()

    device = torch.cuda.is_available()
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load training data
    train_loader = get_dataloader(
        args.batch_size, args.img_size, args.data_dir)

    # Build model

    D, G = build_network(args.conv_dim, args.conv_dim, z_size=args.z_size)

    # Create optimizers for discriminator and generator
    d_optimizer = optim.Adam(D.parameters(), args.lr, [args.beta1, args.beta2])
    g_optimizer = optim.Adam(G.parameters(), args.lr, [args.beta1, args.beta2])

    # Train
    G = train(D, G, args.z_size, train_loader, args.epochs,
              d_optimizer, g_optimizer, device)

    # Save model parameters
    G_path = os.path.join(args.model_dir, 'generator_model_main.pt')
    print('model_dir:', model_dir)
    with open(G_path, 'wb') as f:
        torch.save(G.cpu().state_dict(), f)
