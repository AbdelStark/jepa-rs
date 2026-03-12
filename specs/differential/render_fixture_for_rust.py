#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def emit_tensor(kind: str, name: str, tensor: dict) -> None:
    shape = ",".join(str(value) for value in tensor["shape"])
    values = ",".join(str(value) for value in tensor["values"])
    print(f"{kind}\t{name}\t{shape}\t{values}")


def emit_scalar(name: str, value) -> None:
    print(f"scalar\t{name}\t{value}")


def emit_usizes(name: str, values: list[int]) -> None:
    print(f"usizes\t{name}\t{','.join(str(value) for value in values)}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: render_fixture_for_rust.py /path/to/fixture.json")

    fixture_path = Path(sys.argv[1])
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    emit_scalar("metadata.abs_tolerance", payload["metadata"]["abs_tolerance"])
    emit_scalar("metadata.rel_tolerance", payload["metadata"]["rel_tolerance"])

    encoder = payload["config"]["encoder"]
    predictor = payload["config"]["predictor"]

    for key in (
        "in_channels",
        "image_height",
        "image_width",
        "embed_dim",
        "num_layers",
        "num_heads",
        "mlp_dim",
        "dropout",
    ):
        emit_scalar(f"config.encoder.{key}", encoder[key])
    emit_usizes("config.encoder.patch_size", encoder["patch_size"])

    for key in (
        "encoder_embed_dim",
        "predictor_embed_dim",
        "num_layers",
        "num_heads",
        "max_target_len",
    ):
        emit_scalar(f"config.predictor.{key}", predictor[key])

    emit_tensor("tensor", "raw_input", payload["raw_input"])
    emit_usizes("mask.context_indices", payload["mask"]["context_indices"])
    emit_usizes("mask.target_indices", payload["mask"]["target_indices"])
    emit_scalar("mask.total_tokens", payload["mask"]["total_tokens"])
    emit_usizes("target_positions", payload["target_positions"])
    emit_tensor("tensor", "context", payload["context"])
    emit_tensor("tensor", "target", payload["target"])
    emit_tensor("tensor", "predicted", payload["predicted"])
    print(f"floatlist\tenergy\t{','.join(str(value) for value in payload['energy'])}")

    for name, tensor in sorted(payload["weights"].items()):
        emit_tensor("weight", name, tensor)


if __name__ == "__main__":
    main()
