#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float64) @ weight.astype(np.float64)) + bias.astype(np.float64)).astype(
        np.float32
    )


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + epsilon)
    return normalized * gamma + beta


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def patchify(image: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    batch, channels, height, width = image.shape
    grid_h = height // patch_h
    grid_w = width // patch_w
    patches = []
    for batch_index in range(batch):
        batch_patches = []
        for row in range(grid_h):
            for col in range(grid_w):
                patch = image[
                    batch_index,
                    :,
                    row * patch_h : (row + 1) * patch_h,
                    col * patch_w : (col + 1) * patch_w,
                ]
                batch_patches.append(patch.reshape(-1))
        patches.append(batch_patches)
    return np.asarray(patches, dtype=np.float32)


def rope2d(x: np.ndarray, height: int, width: int) -> np.ndarray:
    batch, seq_len, embed_dim = x.shape
    half_dim = embed_dim // 2
    quarter_dim = half_dim // 2
    cos = np.zeros((seq_len, half_dim), dtype=np.float32)
    sin = np.zeros((seq_len, half_dim), dtype=np.float32)

    freqs = []
    for index in range(quarter_dim):
        freq = 1.0 / (10000.0 ** (2.0 * index / half_dim))
        freqs.append(freq)

    position = 0
    for row in range(height):
        for col in range(width):
            for index, freq in enumerate(freqs):
                angle = row * freq
                cos[position, index] = math.cos(angle)
                sin[position, index] = math.sin(angle)
            for index, freq in enumerate(freqs):
                angle = col * freq
                cos[position, quarter_dim + index] = math.cos(angle)
                sin[position, quarter_dim + index] = math.sin(angle)
            position += 1

    cos = np.broadcast_to(cos[None, :, :], (batch, seq_len, half_dim))
    sin = np.broadcast_to(sin[None, :, :], (batch, seq_len, half_dim))
    x1 = x[:, :, :half_dim]
    x2 = x[:, :, half_dim:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return np.concatenate([out1, out2], axis=2)


def attention(
    x: np.ndarray,
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray,
    out_weight: np.ndarray,
    out_bias: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    batch, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    qkv = linear(x, qkv_weight, qkv_bias)
    q = qkv[:, :, :embed_dim]
    k = qkv[:, :, embed_dim : 2 * embed_dim]
    v = qkv[:, :, 2 * embed_dim :]

    q = q.reshape(batch, seq_len, num_heads, head_dim).swapaxes(1, 2)
    k = k.reshape(batch, seq_len, num_heads, head_dim).swapaxes(1, 2)
    v = v.reshape(batch, seq_len, num_heads, head_dim).swapaxes(1, 2)

    scores = (q @ np.swapaxes(k, -1, -2)) / math.sqrt(head_dim)
    weights = softmax(scores, axis=3)
    out = weights @ v
    out = out.swapaxes(1, 2).reshape(batch, seq_len, embed_dim)
    return linear(out, out_weight, out_bias)


def mlp(
    x: np.ndarray,
    fc1_weight: np.ndarray,
    fc1_bias: np.ndarray,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
) -> np.ndarray:
    return linear(gelu(linear(x, fc1_weight, fc1_bias)), fc2_weight, fc2_bias)


def transformer_block(x: np.ndarray, block: dict[str, np.ndarray], num_heads: int) -> np.ndarray:
    residual = x
    x_norm = layer_norm(x, block["norm1.gamma"], block["norm1.beta"])
    x = residual + attention(
        x_norm,
        block["attn.qkv.weight"],
        block["attn.qkv.bias"],
        block["attn.out_proj.weight"],
        block["attn.out_proj.bias"],
        num_heads,
    )

    residual = x
    x_norm = layer_norm(x, block["norm2.gamma"], block["norm2.beta"])
    return residual + mlp(
        x_norm,
        block["mlp.fc1.weight"],
        block["mlp.fc1.bias"],
        block["mlp.fc2.weight"],
        block["mlp.fc2.bias"],
    )


def sinusoidal_prediction_tokens(max_target_len: int, embed_dim: int) -> np.ndarray:
    tokens = np.zeros((max_target_len, embed_dim), dtype=np.float32)
    for position in range(max_target_len):
        for dim in range(embed_dim):
            exponent = (2 * (dim // 2)) / embed_dim
            angle = position / (10000.0**exponent)
            tokens[position, dim] = math.sin(angle) if dim % 2 == 0 else math.cos(angle)
    return tokens


def bounded_values(shape: tuple[int, ...], low: float, high: float) -> np.ndarray:
    count = int(np.prod(shape))
    if count == 1:
        return np.asarray([low], dtype=np.float32).reshape(shape)
    values = np.linspace(low, high, num=count, dtype=np.float32)
    return values.reshape(shape)


def tensor_payload(array: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {"shape": list(array.shape), "values": array.reshape(-1).tolist()}


def structured_image(channels: int, height: int, width: int, low: float, high: float) -> np.ndarray:
    return bounded_values((1, channels, height, width), low, high)


def make_block(
    prefix: str, embed_dim: int, mlp_dim: int, base_offset: float
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float] | list[int]]]]:
    arrays = {
        "norm1.gamma": bounded_values((embed_dim,), 0.88 + base_offset, 1.08 + base_offset),
        "norm1.beta": bounded_values((embed_dim,), -0.03 + base_offset, 0.03 + base_offset),
        "attn.qkv.weight": bounded_values(
            (embed_dim, 3 * embed_dim), -0.14 + base_offset, 0.14 + base_offset
        ),
        "attn.qkv.bias": bounded_values((3 * embed_dim,), -0.04 + base_offset, 0.04 + base_offset),
        "attn.out_proj.weight": bounded_values(
            (embed_dim, embed_dim), -0.1 + base_offset, 0.1 + base_offset
        ),
        "attn.out_proj.bias": bounded_values((embed_dim,), -0.025 + base_offset, 0.025 + base_offset),
        "norm2.gamma": bounded_values((embed_dim,), 0.9 + base_offset, 1.1 + base_offset),
        "norm2.beta": bounded_values((embed_dim,), -0.025 + base_offset, 0.025 + base_offset),
        "mlp.fc1.weight": bounded_values((embed_dim, mlp_dim), -0.12 + base_offset, 0.12 + base_offset),
        "mlp.fc1.bias": bounded_values((mlp_dim,), -0.03 + base_offset, 0.03 + base_offset),
        "mlp.fc2.weight": bounded_values((mlp_dim, embed_dim), -0.09 + base_offset, 0.09 + base_offset),
        "mlp.fc2.bias": bounded_values((embed_dim,), -0.02 + base_offset, 0.02 + base_offset),
    }
    payload = {
        f"{prefix}.{name}": tensor_payload(array)
        for name, array in arrays.items()
    }
    return arrays, payload


def make_encoder_weights(
    prefix: str,
    *,
    in_channels: int,
    patch_size: tuple[int, int],
    embed_dim: int,
    mlp_dim: int,
    num_layers: int,
    base_offset: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float] | list[int]]]]:
    patch_dim = in_channels * patch_size[0] * patch_size[1]
    weights = {
        "patch_embed.projection.weight": bounded_values(
            (patch_dim, embed_dim), -0.12 + base_offset, 0.12 + base_offset
        ),
        "patch_embed.projection.bias": bounded_values(
            (embed_dim,), -0.03 + base_offset, 0.03 + base_offset
        ),
        "norm.gamma": bounded_values((embed_dim,), 0.94 + base_offset, 1.06 + base_offset),
        "norm.beta": bounded_values((embed_dim,), -0.025 + base_offset, 0.025 + base_offset),
        "blocks": [],
    }

    payload = {
        f"{prefix}.patch_embed.projection.weight": tensor_payload(weights["patch_embed.projection.weight"]),
        f"{prefix}.patch_embed.projection.bias": tensor_payload(weights["patch_embed.projection.bias"]),
        f"{prefix}.norm.gamma": tensor_payload(weights["norm.gamma"]),
        f"{prefix}.norm.beta": tensor_payload(weights["norm.beta"]),
    }

    for layer_index in range(num_layers):
        block, block_payload = make_block(
            f"{prefix}.blocks.{layer_index}",
            embed_dim,
            mlp_dim,
            base_offset + layer_index * 0.01,
        )
        weights["blocks"].append(block)
        payload.update(block_payload)

    return weights, payload


def make_predictor_weights(
    *,
    encoder_embed_dim: int,
    predictor_embed_dim: int,
    num_layers: int,
    base_offset: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float] | list[int]]]]:
    predictor_mlp_dim = predictor_embed_dim * 4
    weights = {
        "input_proj.weight": bounded_values(
            (encoder_embed_dim, predictor_embed_dim), -0.1 + base_offset, 0.1 + base_offset
        ),
        "input_proj.bias": bounded_values(
            (predictor_embed_dim,), -0.025 + base_offset, 0.025 + base_offset
        ),
        "norm.gamma": bounded_values(
            (predictor_embed_dim,), 0.95 + base_offset, 1.05 + base_offset
        ),
        "norm.beta": bounded_values(
            (predictor_embed_dim,), -0.02 + base_offset, 0.02 + base_offset
        ),
        "output_proj.weight": bounded_values(
            (predictor_embed_dim, encoder_embed_dim), -0.09 + base_offset, 0.09 + base_offset
        ),
        "output_proj.bias": bounded_values(
            (encoder_embed_dim,), -0.02 + base_offset, 0.02 + base_offset
        ),
        "blocks": [],
    }

    payload = {
        "predictor.input_proj.weight": tensor_payload(weights["input_proj.weight"]),
        "predictor.input_proj.bias": tensor_payload(weights["input_proj.bias"]),
        "predictor.norm.gamma": tensor_payload(weights["norm.gamma"]),
        "predictor.norm.beta": tensor_payload(weights["norm.beta"]),
        "predictor.output_proj.weight": tensor_payload(weights["output_proj.weight"]),
        "predictor.output_proj.bias": tensor_payload(weights["output_proj.bias"]),
    }

    for layer_index in range(num_layers):
        block, block_payload = make_block(
            f"predictor.blocks.{layer_index}",
            predictor_embed_dim,
            predictor_mlp_dim,
            base_offset + layer_index * 0.0125,
        )
        weights["blocks"].append(block)
        payload.update(block_payload)

    return weights, payload


def encode_image(
    image: np.ndarray,
    encoder: dict[str, np.ndarray],
    encoder_config: dict[str, int | float | list[int]],
    visible_indices: list[int] | None,
) -> np.ndarray:
    patch_h, patch_w = encoder_config["patch_size"]
    grid_h = encoder_config["image_height"] // patch_h
    grid_w = encoder_config["image_width"] // patch_w
    patches = patchify(image, patch_h=patch_h, patch_w=patch_w)
    positioned = rope2d(
        linear(
            patches,
            encoder["patch_embed.projection.weight"],
            encoder["patch_embed.projection.bias"],
        ),
        height=grid_h,
        width=grid_w,
    )
    if visible_indices is not None:
        positioned = positioned[:, visible_indices, :]
    x = positioned
    for block in encoder["blocks"]:
        x = transformer_block(x, block, num_heads=encoder_config["num_heads"])
    return layer_norm(x, encoder["norm.gamma"], encoder["norm.beta"])


def predict(
    context: np.ndarray,
    positions: np.ndarray,
    predictor: dict[str, np.ndarray],
    predictor_config: dict[str, int | float],
) -> np.ndarray:
    batch = context.shape[0]
    pred_tokens = sinusoidal_prediction_tokens(
        predictor_config["max_target_len"],
        predictor_config["predictor_embed_dim"],
    )[positions]
    pred_tokens = np.broadcast_to(
        pred_tokens[None, :, :],
        (batch, positions.shape[0], predictor_config["predictor_embed_dim"]),
    )
    combined = np.concatenate(
        [
            linear(context, predictor["input_proj.weight"], predictor["input_proj.bias"]),
            pred_tokens,
        ],
        axis=1,
    )
    x = combined
    for block in predictor["blocks"]:
        x = transformer_block(x, block, num_heads=predictor_config["num_heads"])
    prediction_slice = x[:, -positions.shape[0] :, :]
    prediction_slice = layer_norm(
        prediction_slice,
        predictor["norm.gamma"],
        predictor["norm.beta"],
    )
    return linear(
        prediction_slice,
        predictor["output_proj.weight"],
        predictor["output_proj.bias"],
    )


FIXTURE_CASES = [
    {
        "filename": "ijepa_strict_tiny_fixture.json",
        "name": "ijepa-strict-image-tiny-v2",
        "reference_note": "Tiny strict image-path parity fixture covering a 2x2 patch grid and single-layer encoder/predictor stack.",
        "abs_tolerance": 1e-5,
        "rel_tolerance": 1e-5,
        "encoder": {
            "in_channels": 1,
            "image_height": 4,
            "image_width": 4,
            "patch_size": [2, 2],
            "embed_dim": 4,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_dim": 8,
            "dropout": 0.0,
        },
        "predictor": {
            "encoder_embed_dim": 4,
            "predictor_embed_dim": 4,
            "num_layers": 1,
            "num_heads": 2,
            "max_target_len": 4,
        },
        "image": structured_image(1, 4, 4, 0.1, 1.6),
        "mask": {"context_indices": [0, 3], "target_indices": [1, 2], "total_tokens": 4},
        "context_base_offset": -0.02,
        "target_base_offset": 0.02,
        "predictor_base_offset": -0.01,
    },
    {
        "filename": "ijepa_strict_rect_fixture.json",
        "name": "ijepa-strict-image-rect-v1",
        "reference_note": "Strict image-path parity fixture covering a non-square 3x4 patch grid with a two-layer encoder and two-layer predictor.",
        "abs_tolerance": 2e-5,
        "rel_tolerance": 2e-5,
        "encoder": {
            "in_channels": 1,
            "image_height": 6,
            "image_width": 8,
            "patch_size": [2, 2],
            "embed_dim": 8,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_dim": 16,
            "dropout": 0.0,
        },
        "predictor": {
            "encoder_embed_dim": 8,
            "predictor_embed_dim": 8,
            "num_layers": 2,
            "num_heads": 2,
            "max_target_len": 12,
        },
        "image": structured_image(1, 6, 8, -0.35, 1.45),
        "mask": {
            "context_indices": [0, 1, 3, 4, 7, 9, 11],
            "target_indices": [2, 5, 6, 8, 10],
            "total_tokens": 12,
        },
        "context_base_offset": -0.015,
        "target_base_offset": 0.025,
        "predictor_base_offset": 0.005,
    },
    {
        "filename": "ijepa_strict_rgb_patch24_fixture.json",
        "name": "ijepa-strict-image-rgb-patch24-v1",
        "reference_note": "Strict image-path parity fixture covering RGB input, asymmetric 2x4 patches, and a predictor that projects from 8 encoder dims into 12 predictor dims.",
        "abs_tolerance": 2e-5,
        "rel_tolerance": 2e-5,
        "encoder": {
            "in_channels": 3,
            "image_height": 8,
            "image_width": 8,
            "patch_size": [2, 4],
            "embed_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_dim": 24,
            "dropout": 0.0,
        },
        "predictor": {
            "encoder_embed_dim": 8,
            "predictor_embed_dim": 12,
            "num_layers": 1,
            "num_heads": 3,
            "max_target_len": 8,
        },
        "image": structured_image(3, 8, 8, -0.6, 1.2),
        "mask": {
            "context_indices": [0, 2, 3, 5, 6],
            "target_indices": [1, 4, 7],
            "total_tokens": 8,
        },
        "context_base_offset": -0.01,
        "target_base_offset": 0.03,
        "predictor_base_offset": -0.005,
    },
]


def build_fixture(case: dict[str, object]) -> dict[str, object]:
    encoder_config = case["encoder"]
    predictor_config = case["predictor"]
    image = case["image"]
    mask = case["mask"]

    context_encoder, context_payload = make_encoder_weights(
        "context_encoder",
        in_channels=encoder_config["in_channels"],
        patch_size=tuple(encoder_config["patch_size"]),
        embed_dim=encoder_config["embed_dim"],
        mlp_dim=encoder_config["mlp_dim"],
        num_layers=encoder_config["num_layers"],
        base_offset=case["context_base_offset"],
    )
    target_encoder, target_payload = make_encoder_weights(
        "target_encoder",
        in_channels=encoder_config["in_channels"],
        patch_size=tuple(encoder_config["patch_size"]),
        embed_dim=encoder_config["embed_dim"],
        mlp_dim=encoder_config["mlp_dim"],
        num_layers=encoder_config["num_layers"],
        base_offset=case["target_base_offset"],
    )
    predictor_weights, predictor_payload = make_predictor_weights(
        encoder_embed_dim=predictor_config["encoder_embed_dim"],
        predictor_embed_dim=predictor_config["predictor_embed_dim"],
        num_layers=predictor_config["num_layers"],
        base_offset=case["predictor_base_offset"],
    )

    target_positions = np.asarray(mask["target_indices"], dtype=np.int64)
    context = encode_image(image, context_encoder, encoder_config, mask["context_indices"])
    target_full = encode_image(image, target_encoder, encoder_config, None)
    target = target_full[:, mask["target_indices"], :]
    predicted = predict(context, target_positions, predictor_weights, predictor_config)
    energy = np.mean((predicted - target) ** 2, axis=(0, 1, 2), keepdims=False).astype(np.float32)

    return {
        "metadata": {
            "name": case["name"],
            "generator": "specs/differential/export_ijepa_strict_fixture.py",
            "reference_repo": "facebookresearch/ijepa",
            "reference_note": case["reference_note"],
            "abs_tolerance": case["abs_tolerance"],
            "rel_tolerance": case["rel_tolerance"],
        },
        "config": {
            "encoder": encoder_config,
            "predictor": predictor_config,
        },
        "weights": {
            **context_payload,
            **target_payload,
            **predictor_payload,
        },
        "raw_input": tensor_payload(image),
        "mask": mask,
        "target_positions": target_positions.tolist(),
        "context": tensor_payload(context),
        "target": tensor_payload(target),
        "predicted": tensor_payload(predicted),
        "energy": [float(energy)],
    }


def main() -> None:
    for case in FIXTURE_CASES:
        fixture_path = OUTPUT_DIR / case["filename"]
        fixture = build_fixture(case)
        fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
