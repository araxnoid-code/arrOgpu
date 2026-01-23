use crate::{ArrayCompute, GpuArrayView};

pub trait ViewArray {}

impl<'a, A> ViewArray for GpuArrayView<'a, A> where A: ArrayCompute {}
