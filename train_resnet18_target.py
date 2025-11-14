import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import math
import matplotlib.pyplot as plt

from dataset import get_train_val_test_loaders
from model.resnet18 import Resnet18
from train_commoncuda_transfer import count_parameters, restore_checkpoint, evaluate_epoch, train_epoch, save_checkpoint, early_stopping
from utils import config, make_training_plot, set_random_seed



def main():
    """Train Resnet18 and show training plots."""
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("resnet18_target.batch_size"),
    )
    datatype = None
    X0, y0 = next(iter(tr_loader))
    datatype = X0.dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Resnet18(num_classes=2)
    model = model.to(device)
    print("model on:", next(model.parameters()).device) 

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=config("resnet18_target.learning_rate"), betas=(0.9, 0.999), weight_decay=0.05
)
    
    print(f"Number of float-valued parameters: {count_parameters(model)}")
    


    steps_per_epoch = len(tr_loader)
    total_epochs = 100               
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
    print("Loading Resnet18...")
    model, start_epoch, stats = restore_checkpoint(model, config("resnet18_target.checkpoint"))
    model = model.to(device)
    print("model on:", next(model.parameters()).device) 
    
    start_epoch = 0
    stats = []    

    axes = make_training_plot(name="resnet18_target Training")

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
        multiclass=False,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    patience = 20
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
            multiclass=False,
        )

        save_checkpoint(model, epoch + 1, config("resnet18_target.checkpoint"), stats)

        # Update early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")

    # Save figure and keep plot open; for debugging
    plt.savefig(f"resnet18_target_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
