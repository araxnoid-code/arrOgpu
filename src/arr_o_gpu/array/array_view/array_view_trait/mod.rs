use crate::{ GpuArray, GpuArrayView };

trait ArrayView {}

impl ArrayView for GpuArray {}

impl<'a> ArrayView for GpuArrayView<'a> {}
