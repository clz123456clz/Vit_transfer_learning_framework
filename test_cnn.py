"""
EECS 445 - Introduction to Machine Learning
Fall 2024  - Project 2

Test CNN
    Test our trained CNN from train_cnn.py on the heldout test data.
    Load the trained CNN model from a saved checkpoint and evaulates using
    accuracy and AUROC metrics.
    Usage: python test_cnn.py
"""

import torch

from dataset import get_train_val_test_loaders
from model.target import Target
from train_common import restore_checkpoint, evaluate_epoch
from utils import config, set_random_seed, make_training_plot

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Print performance metrics for model at specified epoch."""
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("target.batch_size"),
    )

    # Model
    model = Target()

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading cnn...")
    model, start_epoch, stats = restore_checkpoint(model, config("target.checkpoint"))

    axes = make_training_plot()
    print(f"Total learnable parameters: {count_parameters(model)}")

    # Evaluate the model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
    )


if __name__ == "__main__":
    main()
