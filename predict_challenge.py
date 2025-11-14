"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Predict Challenge
    Runs the challenge model inference on the test dataset and saves the
    predictions to disk
    Usage: python predict_challenge.py --uniqname=<uniqname> [--gpu]
"""

import argparse

import pandas as pd
import torch
from torch.nn.functional import softmax
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from dataset_challenge import get_challenge, get_train_val_test_loaders
from model.challenge import Challenge_transfer
from train_commoncuda import restore_checkpoint
from utils import config


def predict_challenge(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device,) -> torch.Tensor:
    """Runs the model inference on the test set and outputs the predictions."""
    y_score = []
    for X, y in data_loader:
        X = X.to(device, non_blocking=True)
        output = model(X)
        probs = softmax(output, dim=1)[:, 1]   
        y_score.append(probs.detach().cpu())
    return torch.cat(y_score)


def main(uniqname: str, gpu: bool) -> None:
    """Train challenge model."""
    # data loaders
    ch_loader, get_semantic_label = get_challenge(
        task="target",
        batch_size=config("challenge.batch_size"),
    )

    tr_loader, _, _, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("adamWvit_transfer_image_aug_mlp_only.batch_size"),
    )
    datatype = None
    X0, y0 = next(iter(tr_loader))
    datatype = X0.dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Challenge_transfer(16, 8, 512, 8, device=device, datatype=datatype, num_classes=2).to(device)

    print("Best epoch:", config("challenge.best_epoch"))

    # Attempts to restore the latest checkpoint if exists
    model, _, _ = restore_checkpoint(model, config("challenge.checkpoint"))
    model.to(device)
    model.eval()

    # Evaluate model
    model_pred = predict_challenge(ch_loader, model, device)

    print("Saving challenge predictions...")

    pd_writer = pd.DataFrame(model_pred, columns=["predictions_gpu" if gpu else "predictions"])
    pd_writer.to_csv(f"{uniqname}.csv", index=False,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniqname", required=True, help="needed to identify your submission")
    parser.add_argument("--gpu", action="store_true", help="only pass this flag if you trained your model using a GPU")
    args = parser.parse_args()
    main(args.uniqname, args.gpu)
