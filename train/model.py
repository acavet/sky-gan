# Model from: https://arxiv.org/pdf/1511.06434.pdf
# Generator architecture: https://www.researchgate.net/profile/Hamed-Alqahtani/publication/338050169/figure/fig10/AS:849390331756554@1579521844631/depicts-the-DCGAN-generator-for-LSUN-sample-scene-modeling-The-DCGAN-models-performance.ppm
# Code based off https://github.com/bvshyam/facegeneration_gan_sagemaker/blob/master/train/model.py

import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernal_size=3, stride=2, padding=0, batch_norm=False):

    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernal_size, stride=stride, padding=padding, bias=False))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def dconv(in_channels, out_channels, kernal_size=3, stride=2, padding=0, batch_norm=False):

    layers = []

    layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                  kernel_size=kernal_size, stride=stride, padding=padding))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


# # FOR 32x32 IMAGES -----------------------------------------------------------------

# class DataDiscriminator(nn.Module):

#     def __init__(self, conv_dim):
#         """
#         Initialize the Discriminator Module
#         :param conv_dim: The depth of the first convolutional layer
#         """
#         super(DataDiscriminator, self).__init__()

#         self.conv_dim = conv_dim

#         # Out - 16
#         self.conv1 = conv(3, conv_dim, kernal_size=4,
#                           padding=1, batch_norm=False)

#         # Out - 8
#         self.conv2 = conv(conv_dim, conv_dim*2, kernal_size=4,
#                           padding=1, batch_norm=True)

#         # Out - 4
#         self.conv3 = conv(conv_dim*2, conv_dim*4,
#                           kernal_size=4, padding=1, batch_norm=True)

#         # Out - 2
#         self.conv4 = conv(conv_dim*4, conv_dim*8,
#                           kernal_size=4, padding=1, batch_norm=True)

#         # Out - 1
#         self.conv5 = conv(conv_dim*8, conv_dim*16,
#                           kernal_size=4, padding=1, batch_norm=False)

#         self.fc = nn.Linear(conv_dim*16, 1)

#     def forward(self, x):
#         """
#         Forward propagation of the neural network
#         :param x: The input to the neural network
#         :return: Discriminator logits; the output of the neural network
#         """
#         # Feedforward behavior
#         out = F.leaky_relu(self.conv1(x), 0.2)
#         out = F.leaky_relu(self.conv2(out), 0.2)
#         out = F.leaky_relu(self.conv3(out), 0.2)
#         out = F.leaky_relu(self.conv4(out), 0.2)
#         out = F.leaky_relu(self.conv5(out), 0.2)
#         out = out.view(-1, self.conv_dim*16)

#         out = self.fc(out)

#         return out

# class DataGenerator(nn.Module):

#     def __init__(self, z_size, conv_dim):
#         """
#         Initialize the Generator Module
#         :param z_size: The length of the input latent vector, z
#         :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
#         """
#         super(DataGenerator, self).__init__()

#         self.z_size = z_size

#         self.conv_dim = conv_dim

#         self.fc = nn.Linear(z_size, conv_dim*8*2*2)

#         # Out - 4
#         self.dconv1 = dconv(conv_dim*8, conv_dim*4,
#                             kernal_size=4, padding=1, batch_norm=True)

#         # Out - 8
#         self.dconv2 = dconv(conv_dim*4, conv_dim*2,
#                             kernal_size=4, padding=1, batch_norm=True)

#         # Out - 16
#         self.dconv3 = dconv(conv_dim*2, conv_dim,
#                             kernal_size=4, padding=1, batch_norm=True)

#         # Out - 32
#         self.dconv4 = dconv(conv_dim, 3, kernal_size=4,
#                             padding=1, batch_norm=False)

#     def forward(self, x):
#         """
#         Forward propagation of the neural network
#         :param x: The input to the neural network
#         :return: A 32x32x3 Tensor image as output
#         """
#         # Feedforward behavior
#         out = self.fc(x)
#         out = out.view(-1, self.conv_dim*8, 2, 2)

#         out = F.relu(self.dconv1(out))
#         out = F.relu(self.dconv2(out))
#         out = F.relu(self.dconv3(out))
#         out = F.tanh(self.dconv4(out))

#         return out


# FOR 64x64 IMAGES -----------------------------------------------------------------

class DataDiscriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(DataDiscriminator, self).__init__()

        self.conv_dim = conv_dim

        # Out - 32
        self.conv1 = conv(3, conv_dim, kernal_size=4,
                          padding=1, batch_norm=False)

        # Out - 16
        self.conv2 = conv(conv_dim, conv_dim*2, kernal_size=4,
                          padding=1, batch_norm=False)

        # Out - 8
        self.conv3 = conv(conv_dim*2, conv_dim*4, kernal_size=4,
                          padding=1, batch_norm=True)

        # Out - 4
        self.conv4 = conv(conv_dim*4, conv_dim*8,
                          kernal_size=4, padding=1, batch_norm=True)

        # Out - 2
        self.conv5 = conv(conv_dim*8, conv_dim*16,
                          kernal_size=4, padding=1, batch_norm=True)

        # Out - 1
        self.conv6 = conv(conv_dim*16, conv_dim*32,
                          kernal_size=4, padding=1, batch_norm=False)

        self.fc = nn.Linear(conv_dim*32, 1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # Feedforward behavior
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = F.leaky_relu(self.conv4(out), 0.2)
        out = F.leaky_relu(self.conv5(out), 0.2)
        out = F.leaky_relu(self.conv6(out), 0.2)
        out = out.view(-1, self.conv_dim*32)

        out = self.fc(out)

        return out


class DataGenerator(nn.Module):

    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(DataGenerator, self).__init__()

        self.z_size = z_size

        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim*16*2*2)

        # Out - 4
        self.dconv1 = dconv(conv_dim*16, conv_dim*8,
                            kernal_size=4, padding=1, batch_norm=True)

        # Out - 8
        self.dconv2 = dconv(conv_dim*8, conv_dim*4,
                            kernal_size=4, padding=1, batch_norm=True)

        # Out - 16
        self.dconv3 = dconv(conv_dim*4, conv_dim*2,
                            kernal_size=4, padding=1, batch_norm=True)

        # Out - 16
        self.dconv4 = dconv(conv_dim*2, conv_dim,
                            kernal_size=4, padding=1, batch_norm=True)

        # Out - 64
        self.dconv5 = dconv(conv_dim, 3, kernal_size=4,
                            padding=1, batch_norm=False)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # Feedforward behavior
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*16, 2, 2)

        out = F.relu(self.dconv1(out))
        out = F.relu(self.dconv2(out))
        out = F.relu(self.dconv3(out))
        out = F.tanh(self.dconv4(out))
        out = F.tanh(self.dconv5(out))

        return out
