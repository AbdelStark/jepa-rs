//! Short-term memory module for world models.
//!
//! Provides a bounded buffer of recent state representations that a world model
//! can attend to when making predictions. This is essential for multi-step planning
//! where the dynamics model benefits from observing recent trajectory history,
//! not just the current state.
//!
//! The memory acts as a FIFO ring buffer: when full, the oldest representation
//! is evicted to make room for the newest one.

use burn::tensor::backend::Backend;

use jepa_core::types::Representation;

/// Bounded short-term memory buffer for state representations.
///
/// Stores the most recent `capacity` state representations in insertion order.
/// When the buffer is full, inserting a new representation evicts the oldest one.
///
/// # Example
///
/// ```
/// use burn::tensor::Tensor;
/// use burn_ndarray::NdArray;
/// use jepa_core::types::Representation;
/// use jepa_world::ShortTermMemory;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let mut memory: ShortTermMemory<B> = ShortTermMemory::new(4);
/// assert!(memory.is_empty());
///
/// let state = Representation::new(Tensor::ones([1, 8, 32], &device));
/// memory.push(state);
/// assert_eq!(memory.len(), 1);
/// assert!(memory.latest().is_some());
/// ```
pub struct ShortTermMemory<B: Backend> {
    /// Ring buffer storage.
    buffer: Vec<Representation<B>>,
    /// Maximum number of entries.
    capacity: usize,
    /// Index of the next write position (wraps around).
    write_pos: usize,
    /// Number of entries currently stored (saturates at capacity).
    len: usize,
}

impl<B: Backend> ShortTermMemory<B> {
    /// Create a new short-term memory with the given capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "memory capacity must be positive");
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            len: 0,
        }
    }

    /// Push a new state representation into memory.
    ///
    /// If the buffer is full, the oldest entry is evicted.
    pub fn push(&mut self, repr: Representation<B>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(repr);
        } else {
            self.buffer[self.write_pos] = repr;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.len = self.len.saturating_add(1).min(self.capacity);
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Maximum number of entries this buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get entries in chronological order (oldest first).
    ///
    /// Returns a `Vec` because the ring buffer's internal order may not
    /// match chronological order once wrapping has occurred.
    pub fn entries_chronological(&self) -> Vec<&Representation<B>> {
        if self.len < self.capacity {
            // Haven't wrapped yet — buffer order is chronological
            self.buffer.iter().collect()
        } else {
            // Wrapped — oldest entry is at write_pos
            let mut result = Vec::with_capacity(self.len);
            for i in 0..self.len {
                let idx = (self.write_pos + i) % self.capacity;
                result.push(&self.buffer[idx]);
            }
            result
        }
    }

    /// Get the most recent entry, if any.
    pub fn latest(&self) -> Option<&Representation<B>> {
        if self.is_empty() {
            return None;
        }
        let idx = if self.write_pos == 0 {
            self.buffer.len() - 1
        } else {
            self.write_pos - 1
        };
        Some(&self.buffer[idx])
    }

    /// Clear all entries from memory.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
        self.len = 0;
    }
}

/// Errors from memory operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("memory capacity must be positive, got {0}")]
    ZeroCapacity(usize),
    #[error("embed_dim mismatch: memory has {expected} but got {actual}")]
    EmbedDimMismatch { expected: usize, actual: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;
    use proptest::prelude::*;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    fn make_repr(val: f32) -> Representation<TestBackend> {
        Representation::new(Tensor::full([1, 1, 1], val, &device()))
    }

    fn repr_value(repr: &Representation<TestBackend>) -> f32 {
        repr.embeddings
            .clone()
            .flatten::<1>(0, 2)
            .into_scalar()
            .elem()
    }

    #[test]
    fn test_new_memory_is_empty() {
        let mem: ShortTermMemory<TestBackend> = ShortTermMemory::new(4);
        assert!(mem.is_empty());
        assert!(!mem.is_full());
        assert_eq!(mem.len(), 0);
        assert_eq!(mem.capacity(), 4);
        assert!(mem.latest().is_none());
    }

    #[test]
    fn test_push_increments_len() {
        let mut mem = ShortTermMemory::new(4);
        mem.push(make_repr(1.0));
        assert_eq!(mem.len(), 1);
        assert!(!mem.is_empty());
        assert!(!mem.is_full());

        mem.push(make_repr(2.0));
        assert_eq!(mem.len(), 2);
    }

    #[test]
    fn test_push_fills_to_capacity() {
        let mut mem = ShortTermMemory::new(3);
        mem.push(make_repr(1.0));
        mem.push(make_repr(2.0));
        mem.push(make_repr(3.0));
        assert!(mem.is_full());
        assert_eq!(mem.len(), 3);
    }

    #[test]
    fn test_push_evicts_oldest_when_full() {
        let mut mem = ShortTermMemory::new(3);
        mem.push(make_repr(1.0));
        mem.push(make_repr(2.0));
        mem.push(make_repr(3.0));
        // Now push a 4th — should evict the 1.0
        mem.push(make_repr(4.0));

        assert_eq!(mem.len(), 3);
        let entries = mem.entries_chronological();
        assert_eq!(entries.len(), 3);

        let values: Vec<f32> = entries.iter().map(|r| repr_value(r)).collect();
        assert!((values[0] - 2.0).abs() < 1e-6, "oldest should be 2.0");
        assert!((values[1] - 3.0).abs() < 1e-6);
        assert!((values[2] - 4.0).abs() < 1e-6, "newest should be 4.0");
    }

    #[test]
    fn test_latest_returns_most_recent() {
        let mut mem = ShortTermMemory::new(4);
        mem.push(make_repr(1.0));
        mem.push(make_repr(2.0));
        mem.push(make_repr(3.0));

        let latest_val = repr_value(mem.latest().unwrap());
        assert!((latest_val - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_latest_after_wrap() {
        let mut mem = ShortTermMemory::new(2);
        mem.push(make_repr(1.0));
        mem.push(make_repr(2.0));
        mem.push(make_repr(3.0)); // evicts 1.0

        let latest_val = repr_value(mem.latest().unwrap());
        assert!((latest_val - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_chronological_order_before_wrap() {
        let mut mem = ShortTermMemory::new(5);
        for i in 1..=3 {
            mem.push(make_repr(i as f32));
        }

        let entries = mem.entries_chronological();
        let values: Vec<f32> = entries.iter().map(|r| repr_value(r)).collect();
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_chronological_order_after_multiple_wraps() {
        let mut mem = ShortTermMemory::new(3);
        // Push 7 items into capacity-3 buffer
        for i in 1..=7 {
            mem.push(make_repr(i as f32));
        }

        let entries = mem.entries_chronological();
        let values: Vec<f32> = entries.iter().map(|r| repr_value(r)).collect();
        // Should contain 5, 6, 7 in order
        assert_eq!(values.len(), 3);
        assert!((values[0] - 5.0).abs() < 1e-6);
        assert!((values[1] - 6.0).abs() < 1e-6);
        assert!((values[2] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_clear() {
        let mut mem = ShortTermMemory::new(4);
        mem.push(make_repr(1.0));
        mem.push(make_repr(2.0));

        mem.clear();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
        assert!(mem.latest().is_none());
        assert!(mem.entries_chronological().is_empty());
    }

    #[test]
    fn test_capacity_one() {
        let mut mem = ShortTermMemory::new(1);
        mem.push(make_repr(1.0));
        assert!(mem.is_full());
        assert_eq!(mem.len(), 1);

        mem.push(make_repr(2.0));
        assert_eq!(mem.len(), 1);

        let latest_val = repr_value(mem.latest().unwrap());
        assert!((latest_val - 2.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "capacity must be positive")]
    fn test_zero_capacity_panics() {
        let _mem: ShortTermMemory<TestBackend> = ShortTermMemory::new(0);
    }

    proptest! {
        #[test]
        fn prop_len_never_exceeds_capacity(
            capacity in 1usize..20,
            num_pushes in 0usize..50,
        ) {
            let mut mem: ShortTermMemory<TestBackend> = ShortTermMemory::new(capacity);
            for i in 0..num_pushes {
                mem.push(make_repr(i as f32));
            }
            prop_assert!(mem.len() <= capacity);
            prop_assert_eq!(mem.len(), num_pushes.min(capacity));
        }

        #[test]
        fn prop_chronological_order_is_correct(
            capacity in 1usize..10,
            num_pushes in 1usize..30,
        ) {
            let mut mem: ShortTermMemory<TestBackend> = ShortTermMemory::new(capacity);
            for i in 0..num_pushes {
                mem.push(make_repr(i as f32));
            }

            let entries = mem.entries_chronological();
            let values: Vec<f32> = entries.iter().map(|r| repr_value(r)).collect();

            // Values should be monotonically increasing (chronological)
            for window in values.windows(2) {
                prop_assert!(window[1] > window[0],
                    "chronological order violated: {} should be > {}",
                    window[1], window[0]);
            }

            // The last value should be the most recent push
            if let Some(last) = values.last() {
                prop_assert!((*last - (num_pushes - 1) as f32).abs() < 1e-5);
            }
        }

        #[test]
        fn prop_latest_is_most_recent(
            capacity in 1usize..10,
            num_pushes in 1usize..30,
        ) {
            let mut mem: ShortTermMemory<TestBackend> = ShortTermMemory::new(capacity);
            for i in 0..num_pushes {
                mem.push(make_repr(i as f32));
            }

            let latest_val = repr_value(mem.latest().unwrap());
            prop_assert!((latest_val - (num_pushes - 1) as f32).abs() < 1e-5,
                "latest should be {} but got {}", num_pushes - 1, latest_val);
        }
    }
}
