use wgpu::{BindGroup, BindGroupLayout};

use crate::{ArrayCompute, MetadataCompound};

pub struct GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    // metadata
    pub(crate) array: &'a A,
    pub(crate) pointer: (u32, u32),
    pub(crate) shape: Vec<u32>,
    pub(crate) stride: Vec<u32>,
    pub(crate) offset: u32,
    pub(crate) metadata_compound: Option<MetadataCompound>,

    // binding
    pub(crate) binding: Option<(BindGroupLayout, BindGroup)>,
}
