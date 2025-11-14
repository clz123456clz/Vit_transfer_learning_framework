
# python list_learnable.py \
#   --module model.deepvit \
#   --cls ViT \
#   --init '{"num_patches":16,"num_blocks":12,"num_hidden":1024,"num_heads":16,"num_classes":2}'

# python list_learnable.py \
#   --module model.deepvit \
#   --cls ViT \
#   --init '{"num_patches":16,"num_blocks":12,"num_hidden":1024,"num_heads":16,"num_classes":2}' \
#   --checkpoint checkpoints/source/epoch=1.checkpoint.pth.tar

import argparse
import importlib
import json
import torch

def human(n):
    return f"{n:,}"

def main():
    parser = argparse.ArgumentParser(
        description="Print all learnable (requires_grad=True) parameter names of a PyTorch model."
    )
    parser.add_argument("--module", required=True, help="Python module path, e.g. model.deepvit")
    parser.add_argument("--cls", required=True, help="Class name in the module, e.g. ViT or Target")
    parser.add_argument("--init", default="{}", help='JSON of model __init__ kwargs, e.g. \'{"num_patches":16,"num_blocks":12,"num_hidden":1024,"num_heads":16,"num_classes":2}\'')
    parser.add_argument("--checkpoint", default=None, help="(Optional) path to a checkpoint .pth/.pt")
    parser.add_argument("--device", default="cpu", help="cuda or cpu (for loading/printing only)")
    args = parser.parse_args()
    
    mod = importlib.import_module(args.module)
    cls = getattr(mod, args.cls)
    kwargs = json.loads(args.init)
    model = cls(**kwargs)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)  
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[info] loaded checkpoint (strict=False). missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(args.device)

    total_learnable = 0
    print("\n=== Learnable Parameters (requires_grad=True) ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            cnt = p.numel()
            total_learnable += cnt
            print(f"{name:60s} shape={tuple(p.shape):20s} numel={human(cnt)}")

    print("\nTotal learnable parameters:", human(total_learnable))

    total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    print("Total frozen parameters   :", human(total_frozen))
    print("Total parameters (all)    :", human(total_all))

if __name__ == "__main__":
    main()
