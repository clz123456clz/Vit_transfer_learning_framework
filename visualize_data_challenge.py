import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from imageio.v2 import imread
from PIL import Image
import torchvision.transforms as T

from dataset_challenge import resize, ImageStandardizer, DogsDataset, get_train_val_test_loaders
from utils import config, denormalize_image, set_random_seed


def inv_normalize_to_hwc_uint8(x_chw: torch.Tensor, mean, std):
    x = x_chw.detach().cpu().clone()             # [C,H,W], float
    x = x.permute(1, 2, 0).numpy()               # -> [H,W,C] numpy
    mean = np.array(mean).reshape(1, 1, -1)      # [1,1,3]
    std  = np.array(std ).reshape(1, 1, -1)      # [1,1,3]
    x = x * std + mean                         
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)  
    return x


if __name__ == "__main__":
    set_random_seed()
    tr_tmp = DogsDataset("train", "target", transform=None)
    standardizer = ImageStandardizer()
    standardizer.fit(tr_tmp.X)
    image_dim = tr_tmp.X[0].shape[0]
    mean = (standardizer.image_mean.squeeze() / 255).astype(np.float32).tolist()
    std  = (standardizer.image_std.squeeze() / 255).astype(np.float32).tolist()
    train_tf = T.Compose([
        T.RandomResizedCrop(image_dim, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.01, 0.02), ratio=(0.3, 3.3)),
    ])

    metadata = pd.read_csv(config("csv_file"))
    metadata = metadata[metadata["partition"] != "challenge"].reset_index(drop=True) 
    print("Click on the figure to choose new images. Close the figure to exit.")

    N = 4
    fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(2 * N, 2 * 2))

    pad = 3
    axes[0, 0].annotate(
        "Original",
        xy=(0, 0.5),
        xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[0, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )
    axes[1, 0].annotate(
        "Preprocessed",
        xy=(0, 0.5),
        xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[1, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    while True:
        rand_idx = np.random.choice(np.arange(len(metadata)), size=N, replace=False)
        X, y = [], []
        for idx in rand_idx:
            filename = os.path.join(config("image_path"), metadata.loc[idx, "filename"])
            X.append(imread(filename))
            y.append(metadata.loc[idx, "semantic_label"])

        for i, (xi, yi) in enumerate(zip(X, y)):
            axes[0, i].imshow(xi)
            axes[0, i].set_title(yi)

        X_ = []
        for x in X:
            img_pil = Image.fromarray(x)        # HWC -> PIL
            img = train_tf(img_pil)
            X_.append(img)

        for i, (xi, yi) in enumerate(zip(X_, y)):
            xi_show = inv_normalize_to_hwc_uint8(xi, mean, std)
            axes[1, i].imshow(xi_show, interpolation="bicubic")

        fig = plt.gcf()
        plt.savefig("visualize_data_challenge.png")
        print("Saved figure to visualize_data.png.")

        def on_close(event):
            exit()

        fig.canvas.mpl_connect("close_event", on_close)

        if plt.waitforbuttonpress():
            break
