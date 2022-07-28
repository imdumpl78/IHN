import torch
import torch.nn as nn
from utils import *

class CNN_weight(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(CNN_weight, self).__init__()

        outputdim = input_dim
        self.layer0 = nn.Sequential(nn.Conv2d(164, 256, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=256//8, num_channels=256), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=128//8, num_channels=128), nn.ReLU())

        self.layer1 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU()
                                    , nn.MaxPool2d(2))
        outputdim_1 = outputdim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU()
                                    , nn.MaxPool2d(2))
        outputdim_2 = outputdim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(2))
        outputdim_3 = outputdim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.layer4_global = nn.Sequential(nn.Conv2d(input_dim, outputdim-64, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim-64) // 8, num_channels=outputdim-64), nn.ReLU())
        outputdim_final = outputdim

        outputdim = outputdim - 64
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.layer5 = nn.Sequential(nn.Conv2d(outputdim + outputdim_final, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(outputdim + outputdim_3, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer7 = nn.Sequential(nn.Conv2d(outputdim+ outputdim_2, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(outputdim + outputdim_1, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer9 = nn.Sequential(nn.Conv2d(outputdim, 1, 3, padding=1, stride=1), nn.Sigmoid())

        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x4 = self.layer4_global(x4)
        x4 = self.up(x4)
        x5 = self.layer5(torch.cat([x4, x3], dim=1))
        x5 = self.up(x5)
        x6 = self.layer6(torch.cat([x5, x2], dim=1))
        x6 = self.up(x6)
        x7 = self.layer7(torch.cat([x6, x1], dim=1))
        x7 = self.up(x7)
        x8 = self.layer8(torch.cat([x7, x], dim=1))
        weight = self.layer9(x8)
        x = weight*x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)
        return x, weight

class CNN_weight_64(nn.Module):
    def __init__(self, input_dim=128):
        super(CNN_weight_64, self).__init__()

        outputdim = input_dim
        self.layer0 = nn.Sequential(nn.Conv2d(164, outputdim*2, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim*2//8, num_channels=outputdim*2), nn.ReLU(), nn.Conv2d(outputdim*2, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU())

        self.layer1 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU()
                                    , nn.MaxPool2d(2))
        outputdim_1 = outputdim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU()
                                    , nn.MaxPool2d(2))
        outputdim_2 = outputdim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(2))
        outputdim_3 = outputdim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(2))

        outputdim_4 = outputdim
        self.layer4p = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.layer_global = nn.Sequential(nn.Conv2d(input_dim, outputdim//2, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim//2) // 8, num_channels=outputdim//2), nn.ReLU())
        outputdim_final = outputdim

        outputdim = outputdim//2
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.layer5 = nn.Sequential(nn.Conv2d(outputdim + outputdim_final, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(outputdim + outputdim_4, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer7 = nn.Sequential(nn.Conv2d(outputdim+ outputdim_3, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(outputdim + outputdim_2, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer8p = nn.Sequential(nn.Conv2d(outputdim + outputdim_1, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU())
        self.layer9 = nn.Sequential(nn.Conv2d(outputdim, 1, 3, padding=1, stride=1), nn.Sigmoid())

        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer4p(x4)

        x5 = self.layer_global(x5)
        x5 = self.up(x5)
        x6 = self.layer5(torch.cat([x5, x4], dim=1))
        x6 = self.up(x6)
        x7 = self.layer6(torch.cat([x6, x3], dim=1))
        x7 = self.up(x7)
        x8 = self.layer7(torch.cat([x7, x2], dim=1))
        x8 = self.up(x8)
        x9 = self.layer8(torch.cat([x8, x1], dim=1))
        x9 = self.up(x9)
        x10 = self.layer8p(torch.cat([x9, x], dim=1))
        weight = self.layer9(x10)

        x = weight*x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4p(x)
        x = self.layer10(x)
        return x, weight

class CNN_64(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(CNN_64, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        outputdim_final = outputdim
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer10(x)

        return x

class CNN(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim_final = outputdim

        ### global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)

        return x

class GMA(nn.Module):
    def __init__(self, args, sz):
        super().__init__()
        self.args = args

        if sz==32:
            if self.args.weight:
                self.cnn_weight = CNN_weight(128)
            else:
                self.cnn = CNN(128)
            
        if sz==64:
            if self.args.weight:
                self.cnn_weight = CNN_weight_64(80)
            else:
                self.cnn = CNN_64(80)

    def forward(self, corr, flow):      
        if self.args.weight:
            delta_flow, weight = self.cnn_weight(torch.cat((corr, flow), dim=1))
            return delta_flow, weight
        else:
            delta_flow = self.cnn(torch.cat((corr, flow), dim=1))
            return delta_flow



