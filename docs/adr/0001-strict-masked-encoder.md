# ADR-0001: Strict Masked Encoder Semantics

## Status

Accepted on March 12, 2026.

## Context

`jepa-train::JepaComponents::forward_step` is generic over `Encoder<B>`. That
generic boundary is useful for shared orchestration, but it cannot express the
JEPA requirement that hidden target tokens be removed before context self-
attention runs.

Changing the public `Encoder` trait to add masked-input methods would ripple
through every crate in the workspace and force downstream users to update their
implementations. The roadmap explicitly asks for minimal public API churn and
for public trait changes to stay human-gated.

## Decision

Strict masked encoding is implemented with encoder-specific helper methods on
the concrete vision encoders:

- `VitEncoder::forward_visible_tokens`
- `VitVideoEncoder::forward_visible_tokens`

The strict training path is then exposed through model-specific helpers:

- `IJepa::forward_step_strict`
- `VJepa::forward_step_strict`

These helpers apply position encoding on the full token grid first, then gather
the visible tokens, and only then run the transformer blocks. This preserves
real flattened token positions while ensuring hidden tokens cannot influence the
context representation through attention.

## Consequences

Positive:

- No `jepa-core` trait signature change.
- The generic trainer remains available for backend-generic experimentation.
- Strict image and video paths share the same semantic pattern.
- Regression tests can directly prove no leakage for the concrete encoders.

Tradeoffs:

- Strict orchestration is currently modality-specific instead of fully generic.
- The generic trainer remains an approximation and must say so in its docs.
- Future modalities need their own pre-attention masking helper unless a later,
  approved trait extension is introduced.

## Migration

Callers that need strict JEPA semantics should move from:

- `JepaComponents::forward_step`

to:

- `IJepa::forward_step_strict` for image JEPA
- `VJepa::forward_step_strict` for video JEPA

Callers that only need a generic orchestration helper can keep using
`JepaComponents::forward_step`, but they should not rely on it as a strict
masked-encoder implementation.
