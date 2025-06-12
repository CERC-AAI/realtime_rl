import torch
import random
import math
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, tanh
import numpy as np
import torch.nn.functional as F

class NatureCNN(nn.Module):
    def __init__(self,actions=6,in_channels=1,k=2.0):
        super(NatureCNN, self).__init__()
        filters_1 = int(k*16)
        filters_2 = int(k*32)
        full_input = int(k*1568)
        self.conv1 = nn.Conv2d(in_channels, filters_1, (8, 8), stride=4)
        self.conv2 = nn.Conv2d(filters_1, filters_2, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(filters_2, filters_2, (3, 3))
        self.linear1 = nn.Linear(full_input, 512)
        self.linear2 = nn.Linear(512, actions)


    def forward(self, x):
        batchSize = x.size()[0]
        #print(x.size(),batchSize)
        x = relu(self.conv1(x))
        #print(x.size())
        x = relu(self.conv2(x))
        #print(x.size())
        x = relu(self.conv3(x))
        #print(x.size())
        x = x.reshape(batchSize,-1)
        #print(x.size())
        x = relu(self.linear1(x))
        #print(x.size())
        x = self.linear2(x)
        #print(x.size())
        return x

class ImpalaCNN(nn.Module):
    def __init__(self,actions=6,in_channels=1,k=1.0):
        super(ImpalaCNN, self).__init__()
        filters_1 = int(k*16)
        filters_2 = int(k*32)
        full_input = int(k*1568)
        self.conv1_base = nn.Conv2d(in_channels, filters_1, (3, 3), stride=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv11_1 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv11_2 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv12_1 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv12_2 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)

        self.conv2_base = nn.Conv2d(filters_1, filters_2, (3, 3), stride=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)
        self.conv21_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv21_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv22_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv22_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.conv3_base = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)
        self.conv31_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv31_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv32_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv32_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.linear1 = nn.Linear(full_input, 512)
        self.linear2 = nn.Linear(512, actions)


    def forward(self, x):
        batchSize = x.size()[0]

        # convolutional sequence 1 
        x = self.maxpool1(self.conv1_base(x))
        # print("x",x.size())
        # residual block 1 - 1
        h = relu(x)
        h = relu(self.conv11_1(h))
        h = self.conv11_2(h)
        x = x + h
        # print("x",x.size())
        # residual block 1 - 2
        h = relu(x)
        h = relu(self.conv12_1(h))
        h = self.conv12_2(h)
        x = x + h
        # print("x",x.size())

        # convolutional sequence 2
        x = self.maxpool2(self.conv2_base(x))
        # print("x",x.size())
        # residual block 2 - 1
        h = relu(x)
        h = relu(self.conv21_1(h))
        h = self.conv21_2(h)
        x = x + h
        # print("x",x.size())
        # residual block 2 - 2
        h = relu(x)
        h = relu(self.conv22_1(h))
        h = self.conv22_2(h)
        x = x + h
        # print("x",x.size())

        # convolutional sequence 3
        x = self.maxpool3(self.conv3_base(x))
        # print("x",x.size())
        # residual block 3 - 1
        h = relu(x)
        h = relu(self.conv31_1(h))
        h = self.conv31_2(h)
        x = x + h
        # print("x",x.size())
        # residual block 3 - 2
        h = relu(x)
        h = relu(self.conv32_1(h))
        h = self.conv32_2(h)
        x = x + h
        # print("x",x.size())
        x = relu(x)
        x = x.reshape(batchSize,-1)
        # print(x.size())
        x = relu(self.linear1(x))
        # print(x.size())
        x = self.linear2(x)
        # print(x.size())
        return x


class DeepImpalaCNN(nn.Module):
    def __init__(self,actions=6,in_channels=1,k=1.0):
        super(DeepImpalaCNN, self).__init__()
        filters_1 = int(k*16)
        filters_2 = int(k*32)
        full_input = int(k*2048)
        self.conv1_base = nn.Conv2d(in_channels, filters_1, (3, 3), stride=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv11_1 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv11_2 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv12_1 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv12_2 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)

        self.conv2_base = nn.Conv2d(filters_1, filters_2, (3, 3), stride=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.conv21_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv21_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv22_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv22_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.conv3_base = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1)
        self.maxpool3 = nn.MaxPool2d(3, stride=1)
        self.conv31_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv31_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv32_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv32_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.conv4_base = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1)
        self.maxpool4 = nn.MaxPool2d(3, stride=1)
        self.conv41_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv41_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv42_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv42_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.conv5_base = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1)
        self.maxpool5 = nn.MaxPool2d(3, stride=2)
        self.conv51_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv51_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv52_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv52_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.conv6_base = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1)
        self.maxpool6 = nn.MaxPool2d(3, stride=1)
        self.conv61_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv61_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv62_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv62_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.linear1 = nn.Linear(full_input, 512)
        self.linear2 = nn.Linear(512, actions)


    def forward(self, x):
        batchSize = x.size()[0]
        #print(x.size())

        # convolutional sequence 1
        x = self.maxpool1(self.conv1_base(x))
        h = relu(x)
        h = relu(self.conv11_1(h))
        h = self.conv11_2(h)
        x = x + h
        h = relu(x)
        h = relu(self.conv12_1(h))
        h = self.conv12_2(h)
        x = x + h
        #print(x.size())

        # convolutional sequence 2
        x = self.maxpool2(self.conv2_base(x))
        h = relu(x)
        h = relu(self.conv21_1(h))
        h = self.conv21_2(h)
        x = x + h
        h = relu(x)
        h = relu(self.conv22_1(h))
        h = self.conv22_2(h)
        x = x + h
        #print(x.size())

        # convolutional sequence 3
        x = self.maxpool3(self.conv3_base(x))
        h = relu(x)
        h = relu(self.conv31_1(h))
        h = self.conv31_2(h)
        x = x + h
        h = relu(x)
        h = relu(self.conv32_1(h))
        h = self.conv32_2(h)
        x = x + h
        #print(x.size())

        # convolutional sequence 4
        x = self.maxpool4(self.conv4_base(x))
        h = relu(x)
        h = relu(self.conv41_1(h))
        h = self.conv41_2(h)
        x = x + h
        h = relu(x)
        h = relu(self.conv42_1(h))
        h = self.conv42_2(h)
        x = x + h
        #print(x.size())

        # convolutional sequence 5
        x = self.maxpool5(self.conv5_base(x))
        h = relu(x)
        h = relu(self.conv51_1(h))
        h = self.conv51_2(h)
        x = x + h
        h = relu(x)
        h = relu(self.conv52_1(h))
        h = self.conv52_2(h)
        x = x + h
        #print(x.size())

        # convolutional sequence 6
        x = self.maxpool6(self.conv6_base(x))
        h = relu(x)
        h = relu(self.conv61_1(h))
        h = self.conv61_2(h)
        x = x + h
        h = relu(x)
        h = relu(self.conv62_1(h))
        h = self.conv62_2(h)
        x = x + h
        #print(x.size())

        # get final output
        x = relu(x)
        x = x.reshape(batchSize,-1)
        x = relu(self.linear1(x))
        x = self.linear2(x)
        return x


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init  # used to initialize weight_sigma and bias_sigma

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Generate new noise and make epsilon matrices."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise to the parameters to perform the forward step"""
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Generate noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())




class RainbowImpalaCNN(nn.Module):
    def __init__(self,actions=6,in_channels=1,k=1.0):
        super(RainbowImpalaCNN, self).__init__()
        atom_size = 51
        filters_1 = int(k*16)
        filters_2 = int(k*32)
        full_input = int(k*1568)
        self.no_dueling = False
        self.no_noise = False
        self.out_dim = actions
        self.atom_size = atom_size
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.support = torch.linspace(
            0, 200, self.atom_size
        ).to(device)

        self.conv1_base = nn.Conv2d(in_channels, filters_1, (3, 3), stride=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv11_1 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv11_2 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv12_1 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)
        self.conv12_2 = nn.Conv2d(filters_1, filters_1, (3, 3), stride=1, padding=1)

        self.conv2_base = nn.Conv2d(filters_1, filters_2, (3, 3), stride=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)
        self.conv21_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv21_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv22_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv22_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.conv3_base = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)
        self.conv31_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv31_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv32_1 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)
        self.conv32_2 = nn.Conv2d(filters_2, filters_2, (3, 3), stride=1, padding=1)

        self.linear1 = nn.Linear(full_input, 512)
        self.hidden_size = 512


        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.advantage_layer = NoisyLinear(self.hidden_size, actions * atom_size)  # output one distribution per action
        # set value layer
        self.value_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.value_layer = NoisyLinear(self.hidden_size, atom_size)

        if self.no_noise:
            # use linear standard layers
            self.advantage_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.advantage_layer = nn.Linear(self.hidden_size, actions * atom_size)
            self.value_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_layer = nn.Linear(self.hidden_size, atom_size)


    def dist(self, x):
        feature = self.getFeatures(x)

        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        if not self.no_dueling:
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # disable dueling network, ignore value layer and advantage formula
            q_atoms = advantage

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def forward(self, x):
        """Forward method implementation, return one Q-value for each action."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    def getFeatures(self, x):
        batchSize = x.size()[0]

        # convolutional sequence 1
        x = self.maxpool1(self.conv1_base(x))
        # print("x",x.size())
        # residual block 1 - 1
        h = relu(x)
        h = relu(self.conv11_1(h))
        h = self.conv11_2(h)
        x = x + h
        # print("x",x.size())
        # residual block 1 - 2
        h = relu(x)
        h = relu(self.conv12_1(h))
        h = self.conv12_2(h)
        x = x + h
        # print("x",x.size())

        # convolutional sequence 2
        x = self.maxpool2(self.conv2_base(x))
        # print("x",x.size())
        # residual block 2 - 1
        h = relu(x)
        h = relu(self.conv21_1(h))
        h = self.conv21_2(h)
        x = x + h
        # print("x",x.size())
        # residual block 2 - 2
        h = relu(x)
        h = relu(self.conv22_1(h))
        h = self.conv22_2(h)
        x = x + h
        # print("x",x.size())

        # convolutional sequence 3
        x = self.maxpool3(self.conv3_base(x))
        # print("x",x.size())
        # residual block 3 - 1
        h = relu(x)
        h = relu(self.conv31_1(h))
        h = self.conv31_2(h)
        x = x + h
        # print("x",x.size())
        # residual block 3 - 2
        h = relu(x)
        h = relu(self.conv32_1(h))
        h = self.conv32_2(h)
        x = x + h
        # print("x",x.size())
        x = relu(x)
        x = x.reshape(batchSize,-1)
        # print(x.size())
        x = relu(self.linear1(x))
        # print(x.size())
        return x

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
