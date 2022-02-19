from absl import app
from absl import flags
import torch
import numpy as np
import torch.nn as nn
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from utils import save_checkpoint, load_checkpoint, record_evaluation, get_last2d
import torch.optim as optim
import config
from dataset import OxDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from SRgen import DeepGen
from ssim_loss import SSIMLoss
import os
import csv



torch.backends.cudnn.benchmark = True

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", config.LEARNING_RATE, "Set the learning rate of the network")
flags.DEFINE_string("train_dir",config.TRAIN_DIR, "The location of the training directory")
flags.DEFINE_string("val_dir",config.VAL_DIR, "The location of the validation directory")
flags.DEFINE_string("test_dir",config.TEST_DIR, "The location of the testing directory")
flags.DEFINE_string("save_dir",config.SAVE_DIR, "The location of the testing directory")
flags.DEFINE_integer("batch_size", config.BATCH_SIZE, "The size of the batches")
flags.DEFINE_integer("num_epochs", config.NUM_EPOCHS, "The number of epochs")


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, ssim_loss,
):
    
    loop = tqdm(loader, leave=True)

    for idx, (x, y, fname) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            #this metric is dirt, too high, guh
            ssim_l = ssim_loss(y, y_fake)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2 

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def test_function(gen, folder='test'):
    print("Testing beginning...")
    test_dataset = OxDataset(root_dir=FLAGS.val_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKERS)

    loop = tqdm(test_loader, leave=True)

    for idx, (x, y, fname) in enumerate(loop):
        x,y,fname= next(iter(test_loader))
        gen.eval()
        with torch.no_grad():
            x, y =x.to(config.DEVICE), y.to(config.DEVICE) 
            y_fake = gen(x)
            fname = str(fname).strip("(,')")
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            x = x * 0.5 + 0.5
            y = y * 0.5 + 0.5
            save_image(y_fake, folder + '/gen' + f"/gen_{fname}")
            save_image(x, folder + '/lowres' + f"/lowres_{fname}")

            #write to the csv with eval metrics
            with open('radargan_test.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                if (os.stat('radargan_test.csv').st_size == 0):
                    # write the header for low res- ground truth comparison and gen-groundtruth comparison
                    writer.writerow(["File Name", "MSE lr_gt","RMSE lr_gt","PNSR lr_gt","SSIM lr_gt",
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
                data = [fname, mse(y, x), rmse(y, x), psnr(y, x), uqi(y, x), ssim(y, x),
                ergas(y, x), scc(y, x), rase(y, x), sam(y, x), msssim(y, x), vifp(y, x)
                , mse(y, y_fake), rmse(y, y_fake), psnr(y, y_fake), uqi(y, y_fake), ssim(y, y_fake),
                ergas(y, y_fake), scc(y, y_fake), rase(y, y_fake), sam(y, y_fake), msssim(y, y_fake), vifp(y, y_fake)]
                # write the data
                writer.writerow(data)
        gen.train()

def main(argv):
    print(config.DEVICE)
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=FLAGS.learning_rate, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=FLAGS.learning_rate, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    ssim_loss = SSIMLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, FLAGS.learning_rate,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, FLAGS.learning_rate,
        )

    train_dataset = OxDataset(root_dir=FLAGS.train_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = OxDataset(root_dir=FLAGS.val_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range(FLAGS.num_epochs):
        print("Running Epoch ", epoch)
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, ssim_loss
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        record_evaluation(gen, val_loader, epoch, folder="evaluation")
    
    test_function(gen)

if __name__ == "__main__":
    app.run(main)