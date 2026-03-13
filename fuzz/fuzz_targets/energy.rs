#![no_main]

use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use jepa_core::energy::{CosineEnergy, EnergyFn, L2Energy, SmoothL1Energy};
use jepa_core::types::Representation;
use libfuzzer_sys::fuzz_target;

type Backend = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

fuzz_target!(|data: &[u8]| {
    let batch = 1 + usize::from(data.first().copied().unwrap_or(0) % 2);
    let seq_len = 1 + usize::from(data.get(1).copied().unwrap_or(0) % 8);
    let embed_dim = 1 + usize::from(data.get(2).copied().unwrap_or(0) % 8);
    let num_values = batch * seq_len * embed_dim;

    let left: Vec<f32> = (0..num_values)
        .map(|index| f32::from(data.get(3 + index).copied().unwrap_or(index as u8)) / 32.0)
        .collect();
    let right: Vec<f32> = (0..num_values)
        .map(|index| {
            f32::from(
                data.get(3 + num_values + index)
                    .copied()
                    .unwrap_or((index * 7) as u8),
            ) / 32.0
        })
        .collect();

    let a = Representation::new(Tensor::<Backend, 3>::from_floats(
        burn::tensor::TensorData::new(left, [batch, seq_len, embed_dim]),
        &device(),
    ));
    let b = Representation::new(Tensor::<Backend, 3>::from_floats(
        burn::tensor::TensorData::new(right, [batch, seq_len, embed_dim]),
        &device(),
    ));

    let _ = L2Energy.compute(&a, &b);
    let _ = CosineEnergy.compute(&a, &b);
    let _ = SmoothL1Energy::new(1.0).compute(&a, &b);
});
