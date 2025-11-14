"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
    Usage: python dataset.py
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
from imageio.v2 import imread
from PIL import Image
import torchvision.transforms as T

from utils import config, set_random_seed


__all__ = [
    "get_train_val_test_loaders", 
    "get_challenge", 
    "get_train_val_test_datasets", 
    "resize", 
    "ImageStandardizer", 
    "DogsDataset"
]


def get_train_val_test_loaders(task: str, batch_size: int, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader, str]:
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the DogsDataset constructor.
    """
    tr, va, te, _ = get_train_val_test_datasets(task, **kwargs)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader, tr.get_semantic_label


class ImageStandardizer:
    """Standardize a batch of images to mean 0 and variance 1.

    The standardization should be applied separately to each channel.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """

    def __init__(self) -> None:
        """Initialize mean and standard deviations to None."""
        self.image_mean = None
        self.image_std = None

    def fit(self, X: npt.NDArray) -> None:
        """Calculate per-channel mean and standard deviation from dataset X."""
        # TODO: 1(a) - Complete this function
        #[B, C, H, W]
        X = np.asarray(X, dtype=np.float64)
        self.image_mean = X.mean(axis=(0, 1, 2), keepdims=True)
        self.image_std  = X.std(axis=(0, 1, 2), keepdims=True)

    
    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """Return standardized dataset given dataset X."""
        # TODO: 1(a) - Complete this function
        X = np.asarray(X, dtype=np.float64)
        X = (X - self.image_mean) / (self.image_std)
        return X


class DogsDataset(Dataset):
    """Dataset class for dog images."""

    def __init__(self, partition: str, task: str = "target", transform= None) -> None:
        """Read in the necessary data from disk.

        For parts 2 and 3, `task` should be "target".
        For source task of part 4, `task` should be "source".
        """
        super().__init__()
        self.transform = transform

        if partition not in ["train", "val", "test", "challenge"]:
            raise ValueError(f"Partition {partition} does not exist")

        set_random_seed()
        self.partition = partition
        self.task = task
        # Load in all the data we need from disk
        if task == "target" or task == "source":
            self.metadata = pd.read_csv(config("csv_file"))
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(
            zip(
                self.metadata[self.metadata.task == self.task]["numeric_label"],
                self.metadata[self.metadata.task == self.task]["semantic_label"],
            )
        )

    def __len__(self) -> int:
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_np = self.X[idx]                 # numpy, CHW, uint8     
        if self.partition == "challenge":
            label = -1
        else:
            label = int(self.y[idx])
        if self.transform is not None:
            img_np = img_np.transpose(1, 2, 0)
            img_pil = Image.fromarray(img_np)        # HWC -> PIL
            img = self.transform(img_pil)        
        else:
            img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # CHW, [0,1]

        return img, torch.tensor(label, dtype=torch.long)

    def _load_data(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Load a single data partition from file."""
        print(f"loading {self.partition}...")

        df = self.metadata[
            (self.metadata.task == self.task)
            & (self.metadata.partition == self.partition)
        ]

        path = config("image_path")

        X, y = [], []
        for i, row in df.iterrows():
            image = imread(os.path.join(path, row["filename"]))
            X.append(image)
            y.append(row["numeric_label"])
        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label: int) -> str:
        """Return the string representation of the numeric class label.

        (e.g., the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]
    
def get_challenge(task: str, batch_size: int, **kwargs) -> tuple[DataLoader, str]:
    """Return DataLoader for challenge dataset.

    Any keyword arguments are forwarded to the DogsDataset constructor.
    """
    tr_tmp = DogsDataset("train", task, transform=None)
    image_dim = tr_tmp.X[0].shape[0]

    standardizer = ImageStandardizer()
    standardizer.fit(tr_tmp.X)
    mean = (standardizer.image_mean.squeeze() / 255.0).astype(np.float32).tolist()
    std  = (standardizer.image_std.squeeze()  / 255.0).astype(np.float32).tolist()

    eval_tf = T.Compose([
        T.Resize(int(image_dim * 1.15)),
        T.CenterCrop(image_dim),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    tr = DogsDataset("train", task, transform=eval_tf, **kwargs)
    ch = DogsDataset("challenge", task, transform=eval_tf, **kwargs)

    ch.X = ch.X.transpose(0, 3, 1, 2)

    ch_loader = DataLoader(ch, batch_size=batch_size, shuffle=False)
    return ch_loader, tr.get_semantic_label



def get_train_val_test_datasets(task: str = "default", **kwargs) -> tuple[DogsDataset, DogsDataset, DogsDataset, ImageStandardizer]:
    """Return DogsDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr_tmp = DogsDataset("train", task, transform=None)
    image_dim = image_dim = tr_tmp.X[0].shape[0]
    standardizer = ImageStandardizer()
    standardizer.fit(tr_tmp.X)
    mean = (standardizer.image_mean.squeeze() / 255.0).astype(np.float32).tolist()
    std  = (standardizer.image_std.squeeze()  / 255.0).astype(np.float32).tolist()  
    
    train_tf = T.Compose([
        T.RandomResizedCrop(image_dim, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.01, 0.02), ratio=(0.3, 3.3)),
    ])
    eval_tf = T.Compose([
        T.Resize(int(image_dim*1.15)),
        T.CenterCrop(image_dim),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


    tr = DogsDataset("train", task, transform=train_tf, **kwargs)
    va = DogsDataset("val",   task, transform=eval_tf,  **kwargs)
    te = DogsDataset("test",  task, transform=eval_tf,  **kwargs)


    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0, 3, 1, 2)
    va.X = va.X.transpose(0, 3, 1, 2)
    te.X = te.X.transpose(0, 3, 1, 2)

    return tr, va, te, standardizer


def resize(X: npt.NDArray) -> npt.NDArray:
    """Resize the data partition X to the size specified in the config file.

    Use bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    image_dim = config("image_dim")
    image_size = (image_dim, image_dim)
    resized = []
    for i in range(X.shape[0]):
        xi = Image.fromarray(X[i]).resize(image_size, resample=2)
        resized.append(xi)
    resized = [np.asarray(im) for im in resized]

    return resized


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_test_datasets(task="target")
    print(f"Train:\t{len(tr.X)}")
    print(f"Val:\t{len(va.X)}")
    print(f"Test:\t{len(te.X)}")
    print(f"Mean:\t{standardizer.image_mean}")
    print(f"Std:\t{standardizer.image_std}")
