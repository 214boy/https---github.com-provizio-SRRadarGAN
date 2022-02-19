import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import cv2
import torch.nn as nn
from generator import Generator

class SSIMLoss(nn.Module):
    def __init__(self,  weight=None, size_average=True):
        super(SSIMLoss, self).__init__()
 
    def forward(self, img1, img2, val_range=255, window_size=11, window=None, size_average=True, full=False):        
        
        L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

        pad = window_size // 2
        
        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()

        # if window is not provided, init one
        if window is None: 
            real_size = min(window_size, height, width) # window should be atleast 11x11 
            window = create_window(real_size, channel=channels).to(img1.device)
        
        # calculating the mu parameter (locally) for both images using a gaussian filter 
        # calculates the luminosity params
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component 
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

        # Some constants for stability 
        C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03 ) ** 2 

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            ret = ssim_score.mean() 
        else: 
            ret = ssim_score.mean(1).mean(1).mean(1)
        if full:
            return ret, contrast_metric
        
        return ret


def gaussian(window_size, sigma):
    """
    this is used to blur portions of the image, allowing for more structural detail
    to be consumed.

    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window



def test():
    ssim_loss = SSIMLoss()
    inp = torch.randn(16,1,256,256)
    inp2 = torch.randn(16,1,256,256)
    res = ssim_loss(inp, inp2)
    print(res)

if __name__ == "__main__":
    test()
