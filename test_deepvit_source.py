import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

from dataset import get_train_val_test_loaders
from model.deepvit import ViT
from train_commoncuda_transfer import evaluate_epoch, restore_checkpoint
from utils import config, make_training_plot, set_random_seed


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Print performance metrics for model at specified epoch."""
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="source",
        batch_size=config("adamWvit_source.batch_size"),
    )
    datatype = None
    X0, y0 = next(iter(tr_loader))
    datatype = X0.dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    model = ViT(
        16, 8, 1024, 16,
        device=device, 
        datatype=datatype,
        num_classes=8,
    )
    model.to(device)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading ToyVit...")
    model, start_epoch, stats = restore_checkpoint(model, config("adamWvit_source.checkpoint"))
    model.to(device)

    axes = make_training_plot()

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
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
        device=device,
        multiclass=True
    )


if __name__ == "__main__":
    main()
