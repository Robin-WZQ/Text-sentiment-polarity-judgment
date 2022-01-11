import re
import torch
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    """
    CNN network.
    """
    def __init__(self,input_layer):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
                    nn.Conv1d(300,10,padding=2,kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv1d(input_layer,out_channels=1,kernel_size=3)
        )

    def forward(self, x):
        x = self.layer(x)
        for i in range(len(x)):
            if x[i]>0.5:
                x[i]=1
            else:
                x[i]=0

        return x

if __name__ == "__main__":
    input = Variable(torch.randn([10,300]))
    net =  CNN()
    output = net(input)
    print(output)
    #torch.Size([1])