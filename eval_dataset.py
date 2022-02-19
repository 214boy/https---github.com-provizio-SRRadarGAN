import numpy as np
import config
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class OxDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = config.transform_only_input(image=image)["image"]
        
        return input_image

if __name__ == "__main__":

    
    dataset = OxDataset("test/")
    loader = DataLoader(dataset, batch_size=1)
    for x in loader:
        print(x.shape)
        x = x.numpy()
        x = (np.squeeze(x, axis=0) *255).astype(np.uint8)
        x = np.transpose(x)
        x = Image.fromarray(x)
        save_image(x, "x.png")
        import sys

        sys.exit()