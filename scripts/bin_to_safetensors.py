"""Convert pytorch_model.bin -> model.safetensors for a HF checkpoint dir.

Usage:
    python scripts/bin_to_safetensors.py <model_dir>

Writes <model_dir>/model.safetensors and leaves the original .bin in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)

    model_dir = Path(sys.argv[1]).resolve()
    bin_path = model_dir / "pytorch_model.bin"
    out_path = model_dir / "model.safetensors"

    if not bin_path.is_file():
        sys.exit(f"error: {bin_path} not found")

    print(f"[convert] loading {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    # safetensors requires contiguous tensors and rejects shared storage.
    cleaned: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            sys.exit(f"error: key {key!r} is not a tensor ({type(tensor).__name__})")
        cleaned[key] = tensor.detach().contiguous().clone()

    print(f"[convert] writing {out_path} with {len(cleaned)} tensors")
    save_file(cleaned, str(out_path), metadata={"format": "pt"})
    print(f"[convert] done -> {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
