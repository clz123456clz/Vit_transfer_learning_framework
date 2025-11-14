"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""

import torch
import matplotlib.pyplot as plt
import math, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from dataset_challenge import get_train_val_test_loaders
from model.challenge import Challenge, Challenge_transfer
from train_commoncuda import count_parameters, restore_checkpoint, evaluate_epoch, train_epoch, save_checkpoint, early_stopping
from utils import config, set_random_seed, make_training_plot
from torch.utils.data import DataLoader


def freeze_head_only(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    for name, p in model.named_parameters():
        if name.startswith("mlp1."):
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

    assert all(n.startswith("mlp1.") for n,_sh,_k in learnable)
    

def train_transfer(
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
    print("\n=== Training target model ===")
    
    # TODO: 3(e) - define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=config("adamWvit_transfer_image_aug_mlp_only.learning_rate"), betas=(0.9, 0.999),
    weight_decay=0.0,
    )


    print("Loading target model with all transformer block layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = make_training_plot("AdamWvit_data_aug_label_smoothing_head8_0.05_transfer Training")

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
        save_checkpoint_dir = "checkpoints/" + model_name + "/"
        # Save model parameters
        save_checkpoint(model, epoch + 1, save_checkpoint_dir, stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")

    # Keep plot open
    print(f"Saving training plot to target_training_plot_only_mlp.png...")
    plt.savefig(f"adamWvit_transfer_image_aug_label_smoothing_head8_0.05_mlp_only.png", dpi=200)
    plt.ioff()
    plt.show()






def main():
    """Train ViT and show training plots."""
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="source",
        batch_size=config("adamWvit_source_image_aug.batch_size"),
    )
    datatype = None
    X0, y0 = next(iter(tr_loader))
    datatype = X0.dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Challenge(16, 8, 512, 8, device=device, datatype=datatype, num_classes=8)
    model = model.to(device)
    print("model on:", next(model.parameters()).device) 

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=config("adamWvit_source_image_aug.learning_rate"), betas=(0.9, 0.999),
    weight_decay=0.05,
    )
    
    print(f"Number of float-valued parameters: {count_parameters(model)}")
    


    steps_per_epoch = len(tr_loader)
    total_epochs = 50               
    warmup_epochs = 5
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)  
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))       # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # Attempts to restore the latest checkpoint if exists
    print("Loading ViT...")
    model, start_epoch, stats = restore_checkpoint(model, config("adamWvit_source_image_aug.checkpoint"))
    model = model.to(device)
    print("model on:", next(model.parameters()).device) 
    
    start_epoch = 0
    stats = []    

    axes = make_training_plot(name="AdamWvit_source_image_aug_label_smoothing_head8_0.05 Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        device=device,
        multiclass=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    patience = 10
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer, device=device, scheduler=scheduler)

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
            device=device,
            multiclass=True,
        )

        save_checkpoint(model, epoch + 1, config("adamWvit_source_image_aug.checkpoint"), stats)

        # Update early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")

    # Save figure and keep plot open; for debugging
    plt.savefig(f"adamWvit_source_image_aug_label_smoothing_head8_0.05_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()

##/////////////////////////////////////////////////////////////////////////////##

    best_epoch = min(range(len(stats)), key=lambda e: stats[e][1])


    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("adamWvit_transfer_image_aug_mlp_only.batch_size"),
    )
    datatype = None
    X0, y0 = next(iter(tr_loader))
    datatype = X0.dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_none = Challenge_transfer(16, 8, 512, 8, device=device, datatype=datatype, num_classes=2).to(device)
    
    print(f"Best epoch from source(min val loss): {best_epoch}")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none,
        config("adamWvit_source_image_aug.checkpoint"),
        force=True,
        pretrain=True,
    )
    freeze_none.to(device)

    freeze_head_only(freeze_none)
    print("\nAll parameter names:")
    for n,_ in freeze_none.named_parameters():
        print(n)

    report_trainable(freeze_none)

    set_random_seed()
    train_transfer(tr_loader, va_loader, te_loader, freeze_none, "adamWvit_transfer_image_aug_label_smoothing_head8_0.05", device)

##/////////////////////////////////////////////////////////////////////////////##
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
