#![no_main]

use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use jepa_core::types::Representation;
use libfuzzer_sys::fuzz_target;

type Backend = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

fuzz_target!(|data: &[u8]| {
    let batch = 1 + usize::from(data.first().copied().unwrap_or(0) % 3);
    let seq_len = 1 + usize::from(data.get(1).copied().unwrap_or(0) % 16);
    let embed_dim = 1 + usize::from(data.get(2).copied().unwrap_or(0) % 8);
    let num_values = batch * seq_len * embed_dim;

    let values: Vec<f32> = (0..num_values)
        .map(|index| {
            let byte = data.get(3 + index).copied().unwrap_or(index as u8);
            f32::from(byte) / 17.0
        })
        .collect();

    let embeddings = Tensor::<Backend, 3>::from_floats(
        burn::tensor::TensorData::new(values, [batch, seq_len, embed_dim]),
        &device(),
    );
    let mask = Tensor::<Backend, 2>::from_floats(
        burn::tensor::TensorData::new(vec![1.0f32; batch * seq_len], [batch, seq_len]),
        &device(),
    );

    let repr = if data.get(3).copied().unwrap_or(0) % 2 == 0 {
        Representation::new(embeddings)
    } else {
        Representation::with_mask(embeddings, mask)
    };

    let raw_indices = data.get(4).copied().unwrap_or(0) % 16;
    let indices: Vec<usize> = (0..usize::from(raw_indices))
        .map(|index| usize::from(data.get(5 + index).copied().unwrap_or(index as u8)) % seq_len)
        .collect();

    let gathered = repr.gather(&indices);
    assert_eq!(gathered.batch_size(), batch);
    assert_eq!(gathered.seq_len(), indices.len());
    assert_eq!(gathered.embed_dim(), embed_dim);
});
