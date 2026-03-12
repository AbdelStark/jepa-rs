#![no_main]

use jepa_compat::keymap::KeyMapping;
use jepa_compat::onnx::{load_checkpoint_from_bytes as load_onnx_checkpoint_from_bytes, OnnxKeyMap, OnnxModelInfo};
use jepa_compat::safetensors::load_checkpoint_from_bytes as load_safetensors_checkpoint_from_bytes;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let empty_mappings: [KeyMapping; 0] = [];
    let _ = load_safetensors_checkpoint_from_bytes(data, &empty_mappings);
    let _ = OnnxModelInfo::from_bytes(data);
    let _ = load_onnx_checkpoint_from_bytes(data, &OnnxKeyMap::new());
});
