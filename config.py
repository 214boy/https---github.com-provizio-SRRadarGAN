import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/media/eddie/fbf78cce-0ffe-4451-bf6f-0a1bb6673213/az_train/az_25k/train"
VAL_DIR = "/media/eddie/fbf78cce-0ffe-4451-bf6f-0a1bb6673213/az_train/az_25k/val"
TEST_DIR = '/media/eddie/fbf78cce-0ffe-4451-bf6f-0a1bb6673213/az_train/az_25k/test'
SAVE_DIR = '/media/eddie/fbf78cce-0ffe-4451-bf6f-0a1bb6673213/az_train/az_25k/test_results'
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 1
IMAGE_SIZE = 501
CHANNELS_IMG = 1
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 1
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        #A.HorizontalFlip(p=0.5),
        #A.ColorJitter(p=0.2),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)