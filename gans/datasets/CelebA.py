import os

import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CelebA
from torchvision import transforms

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() - 4)

class CelebADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_size: int = 64
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):
        # download
        CelebA(self.hparams.data_dir, split='train', download=True, transform=self.transform)
        CelebA(self.hparams.data_dir, split='test', download=True, transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            CelebA_full = CelebA(self.hparams.data_dir, split='train', transform=self.transform)
            self.CelebA_train, self.CelebA_val = random_split(CelebA_full, [0.8, 0.2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.CelebA_test = CelebA(self.hparams.data_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.CelebA_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.CelebA_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.CelebA_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)