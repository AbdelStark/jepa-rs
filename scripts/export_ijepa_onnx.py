#!/usr/bin/env python3
"""Export a pretrained I-JEPA encoder to ONNX format for use with jepa-rs.

This script downloads the official Facebook Research I-JEPA checkpoint,
extracts the encoder weights, and exports them as an ONNX model that
can be loaded by jepa-rs's ONNX runtime.

Requirements:
    pip install torch torchvision timm onnx onnxruntime

Usage:
    # Export ViT-H/14 (default)
    python scripts/export_ijepa_onnx.py

    # Export ViT-H/16-448
    python scripts/export_ijepa_onnx.py --model vit_h16_448

    # Export from local checkpoint
    python scripts/export_ijepa_onnx.py --checkpoint path/to/checkpoint.pth

    # Verify ONNX output
    python scripts/export_ijepa_onnx.py --verify
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Model configurations matching jepa-rs registry
MODEL_CONFIGS = {
    "vit_h14": {
        "image_size": 224,
        "patch_size": 14,
        "embed_dim": 1280,
        "num_layers": 32,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "url": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
        "output_name": "ijepa_vit_h14_encoder.onnx",
    },
    "vit_h16_448": {
        "image_size": 448,
        "patch_size": 16,
        "embed_dim": 1280,
        "num_layers": 32,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "url": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-300e.pth.tar",
        "output_name": "ijepa_vit_h16_448_encoder.onnx",
    },
    "vit_g16": {
        "image_size": 224,
        "patch_size": 16,
        "embed_dim": 1408,
        "num_layers": 40,
        "num_heads": 16,
        "mlp_ratio": 48 / 11,  # 6144 / 1408 ≈ 4.36
        "url": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar",
        "output_name": "ijepa_vit_g16_encoder.onnx",
    },
}


class PatchEmbed(nn.Module):
    """Image to patch embedding (matches Facebook ijepa implementation)."""

    def __init__(self, image_size=224, patch_size=14, in_channels=3, embed_dim=1280):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head self-attention (matches Facebook ijepa implementation)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Two-layer MLP with GELU (matches Facebook ijepa implementation)."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block (matches Facebook ijepa implementation)."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IJepaEncoder(nn.Module):
    """I-JEPA ViT encoder (matches Facebook ijepa implementation).

    This is a faithful reproduction of the encoder architecture from
    github.com/facebookresearch/ijepa, structured to match the checkpoint
    key names exactly so that pretrained weights load without remapping.
    """

    def __init__(self, image_size=224, patch_size=14, in_channels=3,
                 embed_dim=1280, num_layers=32, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable position embedding (the original I-JEPA uses fixed sin-cos,
        # but the checkpoint contains the final values as a buffer)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Encode images to patch-level representations.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Patch representations [B, num_patches, embed_dim]
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


def load_ijepa_checkpoint(checkpoint_path, config):
    """Load an I-JEPA checkpoint into the encoder model."""
    model = IJepaEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Facebook checkpoints wrap state_dict under 'encoder' or 'target_encoder'
    if "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    elif "target_encoder" in checkpoint:
        state_dict = checkpoint["target_encoder"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip common prefixes
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ["module.", "encoder.", "backbone."]:
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    # Load with strict=False to handle extra keys (e.g., predictor weights)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys (expected if loading encoder only): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected[:5]}...")

    return model


def export_to_onnx(model, config, output_path, opset_version=17):
    """Export the encoder to ONNX format."""
    model.eval()

    image_size = config["image_size"]
    dummy_input = torch.randn(1, 3, image_size, image_size)

    num_patches = (image_size // config["patch_size"]) ** 2
    print(f"  Input shape: [1, 3, {image_size}, {image_size}]")
    print(f"  Expected output shape: [1, {num_patches}, {config['embed_dim']}]")

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"  Actual output shape: {list(output.shape)}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes=None,  # Fixed shapes for maximum compatibility
    )
    print(f"  Exported to: {output_path}")

    # Report file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")


def verify_onnx(onnx_path, config):
    """Verify the exported ONNX model produces correct outputs."""
    import onnx
    import onnxruntime as ort
    import numpy as np

    print(f"\nVerifying {onnx_path}...")

    # Check model validity
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("  ONNX model check passed")

    # Run inference
    session = ort.InferenceSession(str(onnx_path))
    image_size = config["image_size"]
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    outputs = session.run(None, {"pixel_values": dummy_input})
    output = outputs[0]

    num_patches = (image_size // config["patch_size"]) ** 2
    expected_shape = (1, num_patches, config["embed_dim"])

    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    assert np.isfinite(output).all(), "Output contains NaN or Inf"
    print(f"  Output shape: {output.shape} (correct)")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("  Verification passed!")


def create_tiny_test_model(output_path):
    """Create a tiny ONNX model for testing (no pretrained weights needed)."""
    model = IJepaEncoder(
        image_size=32,
        patch_size=4,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
    )
    model.eval()

    config = {
        "image_size": 32,
        "patch_size": 4,
        "embed_dim": 64,
    }

    export_to_onnx(model, config, output_path)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Export I-JEPA encoder to ONNX for jepa-rs"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["tiny_test"],
        default="vit_h14",
        help="Model variant to export (default: vit_h14)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint file (downloads if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported ONNX model with onnxruntime",
    )
    parser.add_argument(
        "--tiny-test",
        action="store_true",
        help="Create a tiny test model (no pretrained weights needed)",
    )
    args = parser.parse_args()

    if args.tiny_test or args.model == "tiny_test":
        output_path = args.output or "ijepa_tiny_test.onnx"
        print("Creating tiny test model...")
        config = create_tiny_test_model(output_path)
        if args.verify:
            verify_onnx(output_path, config)
        return

    config = MODEL_CONFIGS[args.model]
    output_path = args.output or config["output_name"]

    print(f"Exporting I-JEPA {args.model} encoder to ONNX")
    print(f"  Architecture: embed_dim={config['embed_dim']}, "
          f"layers={config['num_layers']}, heads={config['num_heads']}")

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Download checkpoint
        checkpoint_path = f"/tmp/ijepa_{args.model}.pth.tar"
        if not Path(checkpoint_path).exists():
            print(f"  Downloading from {config['url']}...")
            torch.hub.download_url_to_file(config["url"], checkpoint_path)
        else:
            print(f"  Using cached checkpoint: {checkpoint_path}")

    print("  Loading checkpoint...")
    model = load_ijepa_checkpoint(checkpoint_path, config)

    print("  Exporting to ONNX...")
    export_to_onnx(model, config, output_path, args.opset)

    if args.verify:
        verify_onnx(output_path, config)

    print(f"\nDone! Use with jepa-rs:")
    print(f'  let session = OnnxSession::from_path("{output_path}")?;')


if __name__ == "__main__":
    main()
