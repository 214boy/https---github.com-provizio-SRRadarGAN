import torch
from utils import save_checkpoint, load_checkpoint, record_evaluation
import torch.nn as nn
import torch.optim as optim
import config
from torch.utils.data import DataLoader
from dataset import OxDataset
from generator import Generator
from torchvision.utils import save_image

def test():
    #get oxford dataset
    dataset = OxDataset(root_dir='/home/eddie/data/testset/train')
    #setup your dataloader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    #setup the generator
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
    record_evaluation(gen,loader,1,'./evaluation/')

if __name__ == "__main__":
    test()