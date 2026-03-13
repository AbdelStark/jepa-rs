//! Shared token-sequence utilities for vision encoders.

use burn::prelude::*;

/// Gather a subset of tokens from a `[batch, seq_len, embed_dim]` tensor by index.
///
/// Validates that all indices are within bounds, returning an empty token
/// sequence when `indices` is empty.
pub(crate) fn gather_token_sequence<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: &[usize],
) -> Tensor<B, 3> {
    let [batch, seq_len, embed_dim] = tokens.dims();
    let device = tokens.device();

    if indices.is_empty() {
        return Tensor::zeros([batch, 0, embed_dim], &device);
    }

    // Validate that all indices are within bounds before calling select(),
    // which may panic or produce undefined results on out-of-range indices.
    for &idx in indices {
        assert!(
            idx < seq_len,
            "gather index {idx} out of bounds for sequence length {seq_len}",
        );
    }

    let index_data: Vec<i64> = indices.iter().map(|&index| index as i64).collect();
    let index_tensor =
        Tensor::<B, 1, Int>::from_data(TensorData::new(index_data, [indices.len()]), &device);

    tokens.select(1, index_tensor)
}
