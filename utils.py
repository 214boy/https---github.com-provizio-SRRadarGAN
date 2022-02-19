import torch
import config
from torchvision.utils import save_image
import os
import csv
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


def record_evaluation(gen, val_loader, epoch, folder):
    x,y,fname= next(iter(val_loader))
    gen.eval()
    with torch.no_grad():
        x, y =x.to(config.DEVICE), y.to(config.DEVICE) 
        y_fake = gen(x)
        fname = str(fname).strip("(,')")
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        save_image(y_fake, folder + '/gen' + f"/gen_{epoch}_{fname}")
        save_image(x, folder + '/lowres' + f"/lowres_{epoch}_{fname}")
        save_image(y, folder + '/gtruth' + f"/gtruth_{epoch}_{fname}")

        #write to the csv with eval metrics
        with open('radargan_eval.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if (os.stat('radargan_eval.csv').st_size == 0):
                # write the header for low res- ground truth comparison and gen-groundtruth comparison
                writer.writerow(["File Name","Epoch" , "MSE lr_gt","RMSE lr_gt","PNSR lr_gt","SSIM lr_gt",
                "UQI lr_gt","MSSSIM lr_gt","ERGAS lr_gt","SCC lr_gt","RASE lr_gt","SAM lr_gt","VIF lr_gt", 
                "MSE gen-gt","RMSE gen-gt","PNSR gen-gt","SSIM gen-gt",
                "UQI gen-gt","MSSSIM gen-gt","ERGAS gen-gt","SCC gen-gt","RASE gen-gt","SAM gen-gt","VIF gen-gt"])
            
            #convert to numpy 2d array
            x =  (get_last2d(x.detach().cpu().numpy()) * 255).astype(int)
            y =  (get_last2d(y.detach().cpu().numpy()) * 255).astype(int)
            y_fake =  (get_last2d(y_fake.detach().cpu().numpy()) * 255).astype(int)
            '''
            Metrics for evaluating image similarity --------------------------
                MSE - Mean Squared Error - pixel by pixel squared difference
                RMSE - Root Mean Squared Error - pixel by pixel squared difference rooted
                PNSR - Peak Signal to Noise Ratio - PSNR = 20 log10(MAX/(MSE)^(1/2)).
                SSIM - Structural Similarity Index Metric - measure of structural similarity, employs gaussian noise using a windown func
                UQI - Universal Quality Index - quality of an image using loss of correlation, luminance distortion, and contrast distortion.
                MSSSIM - Multiscale SSIM - exactly what it says on the tin
                ERGAS - relative dimensionless global error 
                SSC - Self Correlation Coefficient
                RASE - Relative average spectral error
                SAM - spectral angle mapper
                VIF - calculates Pixel Based Visual Information Fidelity (vif-p).
            '''
            #data is collected for the similarity between in low res and ground truth, and the generated and ground truth
            data = [fname, epoch, mse(y, x), rmse(y, x), psnr(y, x), uqi(y, x), ssim(y, x),
             ergas(y, x), scc(y, x), rase(y, x), sam(y, x), msssim(y, x), vifp(y, x)
             , mse(y, y_fake), rmse(y, y_fake), psnr(y, y_fake), uqi(y, y_fake), ssim(y, y_fake),
             ergas(y, y_fake), scc(y, y_fake), rase(y, y_fake), sam(y, y_fake), msssim(y, y_fake), vifp(y, y_fake)]
            # write the data
            writer.writerow(data)
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_last2d(data):
    if data.ndim <= 2:
        return data
    slc = [0] * (data.ndim - 2)
    slc += [slice(None), slice(None)]
    return data[slc]
