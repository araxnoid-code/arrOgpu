use crate::GpuArray;

pub trait ContiguousArray {}

impl ContiguousArray for GpuArray {}
