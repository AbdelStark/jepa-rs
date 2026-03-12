#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np


OUTPUT_PATH = Path(__file__).with_name("ijepa_strict_tiny_fixture.json")


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float64) @ weight.astype(np.float64)) + bias.astype(np.float64)).astype(
        np.float32
    )


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
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


def attention(x: np.ndarray, qkv_weight: np.ndarray, qkv_bias: np.ndarray, out_weight: np.ndarray, out_bias: np.ndarray, num_heads: int) -> np.ndarray:
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


def mlp(x: np.ndarray, fc1_weight: np.ndarray, fc1_bias: np.ndarray, fc2_weight: np.ndarray, fc2_bias: np.ndarray) -> np.ndarray:
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
            angle = position / (10000.0 ** exponent)
            tokens[position, dim] = math.sin(angle) if dim % 2 == 0 else math.cos(angle)
    return tokens


def seq_values(shape: tuple[int, ...], start: float, step: float) -> np.ndarray:
    values = start + step * np.arange(np.prod(shape), dtype=np.float32)
    return values.reshape(shape).astype(np.float32)


def make_block(prefix: str, weight_specs: dict[str, tuple[tuple[int, ...], float, float]]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float] | list[int]]]]:
    arrays = {}
    payload = {}
    for suffix, (shape, start, step) in weight_specs.items():
        name = f"{prefix}.{suffix}"
        array = seq_values(shape, start, step)
        arrays[suffix] = array
        payload[name] = {"shape": list(shape), "values": array.reshape(-1).tolist()}
    return arrays, payload


def make_encoder_weights(prefix: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float] | list[int]]]]:
    block_arrays, block_payload = make_block(
        f"{prefix}.blocks.0",
        {
            "norm1.gamma": ((4,), 0.85, 0.07),
            "norm1.beta": ((4,), -0.03, 0.02),
            "attn.qkv.weight": ((4, 12), -0.24, 0.01),
            "attn.qkv.bias": ((12,), -0.05, 0.01),
            "attn.out_proj.weight": ((4, 4), -0.12, 0.015),
            "attn.out_proj.bias": ((4,), -0.03, 0.02),
            "norm2.gamma": ((4,), 0.9, 0.05),
            "norm2.beta": ((4,), 0.02, -0.015),
            "mlp.fc1.weight": ((4, 8), -0.16, 0.01),
            "mlp.fc1.bias": ((8,), -0.04, 0.01),
            "mlp.fc2.weight": ((8, 4), -0.08, 0.009),
            "mlp.fc2.bias": ((4,), 0.01, 0.015),
        },
    )

    weights = {
        "patch_embed.projection.weight": seq_values((4, 4), -0.18, 0.03),
        "patch_embed.projection.bias": seq_values((4,), -0.06, 0.03),
        "norm.gamma": np.asarray([1.05, 0.95, 1.1, 0.9], dtype=np.float32),
        "norm.beta": np.asarray([0.02, -0.01, 0.03, -0.02], dtype=np.float32),
        "blocks.0": block_arrays,
    }

    payload = {
        f"{prefix}.patch_embed.projection.weight": {
            "shape": [4, 4],
            "values": weights["patch_embed.projection.weight"].reshape(-1).tolist(),
        },
        f"{prefix}.patch_embed.projection.bias": {
            "shape": [4],
            "values": weights["patch_embed.projection.bias"].reshape(-1).tolist(),
        },
        f"{prefix}.norm.gamma": {
            "shape": [4],
            "values": weights["norm.gamma"].tolist(),
        },
        f"{prefix}.norm.beta": {
            "shape": [4],
            "values": weights["norm.beta"].tolist(),
        },
        **block_payload,
    }
    return weights, payload


def make_predictor_weights() -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float] | list[int]]]]:
    block_arrays, block_payload = make_block(
        "predictor.blocks.0",
        {
            "norm1.gamma": ((4,), 1.0, 0.04),
            "norm1.beta": ((4,), 0.01, -0.01),
            "attn.qkv.weight": ((4, 12), -0.14, 0.008),
            "attn.qkv.bias": ((12,), -0.02, 0.005),
            "attn.out_proj.weight": ((4, 4), -0.06, 0.012),
            "attn.out_proj.bias": ((4,), 0.03, -0.01),
            "norm2.gamma": ((4,), 0.95, 0.03),
            "norm2.beta": ((4,), -0.02, 0.01),
            "mlp.fc1.weight": ((4, 16), -0.12, 0.005),
            "mlp.fc1.bias": ((16,), -0.03, 0.004),
            "mlp.fc2.weight": ((16, 4), -0.09, 0.004),
            "mlp.fc2.bias": ((4,), 0.02, 0.01),
        },
    )

    weights = {
        "input_proj.weight": seq_values((4, 4), -0.11, 0.02),
        "input_proj.bias": seq_values((4,), -0.03, 0.015),
        "norm.gamma": np.asarray([0.98, 1.02, 0.96, 1.04], dtype=np.float32),
        "norm.beta": np.asarray([-0.01, 0.02, -0.02, 0.01], dtype=np.float32),
        "output_proj.weight": seq_values((4, 4), -0.09, 0.017),
        "output_proj.bias": seq_values((4,), -0.02, 0.01),
        "blocks.0": block_arrays,
    }

    payload = {
        "predictor.input_proj.weight": {
            "shape": [4, 4],
            "values": weights["input_proj.weight"].reshape(-1).tolist(),
        },
        "predictor.input_proj.bias": {
            "shape": [4],
            "values": weights["input_proj.bias"].tolist(),
        },
        "predictor.norm.gamma": {
            "shape": [4],
            "values": weights["norm.gamma"].tolist(),
        },
        "predictor.norm.beta": {
            "shape": [4],
            "values": weights["norm.beta"].tolist(),
        },
        "predictor.output_proj.weight": {
            "shape": [4, 4],
            "values": weights["output_proj.weight"].reshape(-1).tolist(),
        },
        "predictor.output_proj.bias": {
            "shape": [4],
            "values": weights["output_proj.bias"].tolist(),
        },
        **block_payload,
    }
    return weights, payload


def encode_image(image: np.ndarray, encoder: dict[str, np.ndarray], visible_indices: list[int] | None) -> np.ndarray:
    patches = patchify(image, patch_h=2, patch_w=2)
    positioned = rope2d(
        linear(
            patches,
            encoder["patch_embed.projection.weight"],
            encoder["patch_embed.projection.bias"],
        ),
        height=2,
        width=2,
    )
    if visible_indices is not None:
        positioned = positioned[:, visible_indices, :]
    x = transformer_block(positioned, encoder["blocks.0"], num_heads=2)
    return layer_norm(x, encoder["norm.gamma"], encoder["norm.beta"])


def predict(context: np.ndarray, positions: np.ndarray, predictor: dict[str, np.ndarray]) -> np.ndarray:
    batch = context.shape[0]
    pred_tokens = sinusoidal_prediction_tokens(4, 4)[positions]
    pred_tokens = np.broadcast_to(pred_tokens[None, :, :], (batch, positions.shape[0], 4))
    combined = np.concatenate(
        [
            linear(context, predictor["input_proj.weight"], predictor["input_proj.bias"]),
            pred_tokens,
        ],
        axis=1,
    )
    combined = transformer_block(combined, predictor["blocks.0"], num_heads=2)
    prediction_slice = combined[:, -positions.shape[0] :, :]
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


def main() -> None:
    encoder_config = {
        "in_channels": 1,
        "image_height": 4,
        "image_width": 4,
        "patch_size": [2, 2],
        "embed_dim": 4,
        "num_layers": 1,
        "num_heads": 2,
        "mlp_dim": 8,
        "dropout": 0.0,
    }
    predictor_config = {
        "encoder_embed_dim": 4,
        "predictor_embed_dim": 4,
        "num_layers": 1,
        "num_heads": 2,
        "max_target_len": 4,
    }

    context_encoder, context_payload = make_encoder_weights("context_encoder")
    target_encoder, target_payload = make_encoder_weights("target_encoder")
    predictor_weights, predictor_payload = make_predictor_weights()

    image = np.asarray(
        [
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                    [1.3, 1.4, 1.5, 1.6],
                ]
            ]
        ],
        dtype=np.float32,
    )
    mask = {"context_indices": [0, 3], "target_indices": [1, 2], "total_tokens": 4}
    target_positions = np.asarray(mask["target_indices"], dtype=np.int64)

    context = encode_image(image, context_encoder, mask["context_indices"])
    target_full = encode_image(image, target_encoder, None)
    target = target_full[:, mask["target_indices"], :]
    predicted = predict(context, target_positions, predictor_weights)
    energy = np.mean((predicted - target) ** 2, axis=(0, 1, 2), keepdims=False).astype(np.float32)

    fixture = {
        "metadata": {
            "name": "ijepa-strict-image-tiny-v1",
            "generator": "specs/differential/export_ijepa_strict_fixture.py",
            "reference_repo": "facebookresearch/ijepa",
            "reference_note": "Exports a tiny strict image-path parity fixture using the same flattened token and masked-gather semantics as the canonical Python I-JEPA stack.",
            "abs_tolerance": 1e-5,
            "rel_tolerance": 1e-5,
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
        "raw_input": {
            "shape": [1, 1, 4, 4],
            "values": image.reshape(-1).tolist(),
        },
        "mask": mask,
        "target_positions": target_positions.tolist(),
        "context": {
            "shape": list(context.shape),
            "values": context.reshape(-1).tolist(),
        },
        "target": {
            "shape": list(target.shape),
            "values": target.reshape(-1).tolist(),
        },
        "predicted": {
            "shape": list(predicted.shape),
            "values": predicted.reshape(-1).tolist(),
        },
        "energy": [float(energy)],
    }

    OUTPUT_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
