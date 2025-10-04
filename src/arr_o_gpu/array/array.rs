use std::sync::Arc;

use crate::ArrOgpuModule;

struct GpuArray {
    module: Arc<ArrOgpuModule>,
    pointer: usize,
    length: usize,
    shape: Vec<usize>,
}
