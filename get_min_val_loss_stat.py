from dataset_challenge import get_train_val_test_loaders
from train_common import restore_checkpoint, evaluate_epoch
from utils import config, set_random_seed, make_training_plot
import torch
import os

import os
import torch

def main():
    checkpoint_dir = "checkpoints/adamWvit_transfer_image_aug_label_smoothing_head8_0.05/"

    cp_files = [f for f in os.listdir(checkpoint_dir)
                if f.startswith("epoch=") and f.endswith(".checkpoint.pth.tar")]
    if not cp_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    def _parse_epoch(fname: str) -> int:
        # "epoch=17.checkpoint.pth.tar" -> 17
        return int(fname[len("epoch="):].split(".checkpoint")[0])

    latest_epoch = max(_parse_epoch(f) for f in cp_files)
    print(f"Latest_epoch: {latest_epoch}")
    filename = os.path.join(checkpoint_dir, f"epoch={latest_epoch}.checkpoint.pth.tar")

    ckpt = torch.load(filename, weights_only=False)
    stats = ckpt["stats"]            # list[list[metrics]]
    if not stats or len(stats[0]) < 2:
        raise ValueError("stats does not contain validation loss at index 1.")

    val_losses = [row[1] for row in stats]      # stats[e][1] = val loss
    best_epoch = min(range(len(val_losses)), key=lambda e: val_losses[e])

    def get_or_none(row, idx):
        return row[idx] if len(row) > idx else None

    train_auc = get_or_none(stats[best_epoch], 5)  # train AUROC
    val_auc   = get_or_none(stats[best_epoch], 2)  # val   AUROC
    test_auc  = get_or_none(stats[best_epoch], 8)  # test  AUROC (target/finetune)

    print(f"Best epoch (min val loss): {best_epoch}")
    print(f"Train AUROC: {train_auc}")
    print(f"Val   AUROC: {val_auc}")
    print(f"Test  AUROC: {test_auc: 4f}")

    


if __name__ == "__main__":
    main()