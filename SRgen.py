"""Take the existing U-net generator model and iterate this model to 
allow for better understanding of the images presented"""

from generator import Generator
import torch
import torch.nn as nn

class DeepGen(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(DeepGen, self).__init__()
        self.unet = Generator(in_channels,features)

    def forward(self, x):
        u1 = self.unet(x)
        u2 = self.unet(u1)
        u3 = self.unet(u2)
        u4 = self.unet(u3)
        u5 = self.unet(u4)
        return u5

def test():
    x = torch.randn((1,1,256,256))
    model = DeepGen(in_channels=1, features=64)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()