//! Backend type aliases and device initialization for browser execution.
//!
//! Provides two backend configurations:
//! - `WebBackend`: GPU-accelerated via `Autodiff<Wgpu>` (WebGPU)
//! - `CpuBackend`: CPU fallback via `Autodiff<NdArray<f32>>`
//!
//! At runtime the demo detects WebGPU availability and selects the
//! appropriate backend.

use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;

/// GPU-accelerated backend for WebGPU-capable browsers.
pub type WebBackend = Autodiff<Wgpu>;

/// CPU fallback backend for browsers without WebGPU support.
pub type CpuBackend = Autodiff<NdArray<f32>>;

/// Device handle for the CPU fallback backend.
pub const CPU_DEVICE: burn_ndarray::NdArrayDevice = burn_ndarray::NdArrayDevice::Cpu;

/// Initialize the default WGPU device for use in the browser.
///
/// This creates a `WgpuDevice` that maps to the browser's WebGPU adapter.
pub fn wgpu_device() -> burn_wgpu::WgpuDevice {
    burn_wgpu::WgpuDevice::DefaultDevice
}
