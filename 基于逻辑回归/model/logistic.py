import re
import torch
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Variable


class logistic_net(nn.Module):
    """
    logistic network.
    """
    def __init__(self):
        super(logistic_net, self).__init__()
        self.layer = nn.Sequential(
                    nn.Linear(300,1),
                    nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == "__main__":
    input = Variable(torch.randn([100,300]))
    net =  logistic_net()
    output = net(input)
    print(output)
    #torch.Size([1])