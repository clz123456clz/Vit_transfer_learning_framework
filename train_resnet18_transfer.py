from copy import deepcopy
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import get_train_val_test_loaders
from model.resnet18_transfer import Resnet18
from train_commoncuda_transfer import evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import config, make_training_plot, set_random_seed


__all__ = ["freeze_layers", "train"]



def freeze_head_only(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    for name, p in model.named_parameters():
        if name.startswith("fc_2."):
            p.requires_grad_(True)

def report_trainable(model):
    learnable = [(n, tuple(p.shape), p.numel()) 
                 for n,p in model.named_parameters() if p.requires_grad]
    frozen = [(n, tuple(p.shape), p.numel()) 
              for n,p in model.named_parameters() if not p.requires_grad]

    print("\n=== Trainable params ===")
    for n, sh, k in learnable:
        print(f"{n:60s} shape={str(sh):>18s} numel={k:,}")
    print(f"\nTotal trainable: {sum(k for _,_,k in learnable):,}")
    print(f"Total frozen   : {sum(k for _,_,k in frozen):,}")
    print(f"Total params   : {sum(k for _,_,k in learnable+frozen):,}")

    assert all(n.startswith("fc_2.") for n,_sh,_k in learnable)
    

def train(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    te_loader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    device: str,
) -> None:
    """
    This function trains the target model. Only the weights of unfrozen layers of the model passed 
    into this function will be updated in training.
    
    Args:
        tr_loader: DataLoader for training data
        va_loader: DataLoader for validation data
        te_loader: DataLoader for test data
        model: subclass of torch.nn.Module, model to train on
        model_name: str, checkpoint path for the model
        num_layers: int, the number of source model layers to freeze
    """
    set_random_seed()
    
    # TODO: 3(e) - define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Loading target model with all transformer block layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = make_training_plot("Resnet18_transfer Training")

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
        device=device
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: 3(e) - patience for early stopping, see appendix B 
    patience = 10
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer, device=device)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
            device=device,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")

    # Keep plot open
    print(f"Saving training plot to resnet18_target_training_plot_only_fc_2.png...")
    plt.savefig(f"resnet18_target_training_plot_only_fc_2.png", dpi=200)
    plt.ioff()
    plt.show()


def main() -> None:
    """
    Train transfer learning model and display training plots.

    Train four different models with {0, 1, 2, 3} layers frozen.
    """
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("resnet18_transfer.batch_size"),
    )
    datatype = None
    X0, y0 = next(iter(tr_loader))
    datatype = X0.dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_none = Resnet18(num_classes=2).to(device)
    print("Loading source...")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none,
        config("resnet18_source.checkpoint"),
        force=True,
        pretrain=True,
    )
    freeze_none.to(device)

    freeze_head_only(freeze_none)
    print("\nAll parameter names:")
    for n,_ in freeze_none.named_parameters():
        print(n)

    report_trainable(freeze_none)

    train(tr_loader, va_loader, te_loader, freeze_none, "resnet18_transfer_fc_2_only", device)


if __name__ == "__main__":
    main()
