use wgpu::{BindGroup, BindGroupLayout};

use crate::{ArrayType, CheckArrayType, GpuArray, GpuArrayView, MetadataCompound};

pub trait ArrayCompute {
    fn module(&self) -> &std::sync::Arc<crate::ArrOgpuModule>;

    fn pointer_to_arr(&self) -> [u32; 2];

    fn dim(&self) -> usize;

    fn stride(&self) -> &Vec<u32>;

    fn shape(&self) -> &Vec<u32>;

    fn pointer(&self) -> (u32, u32);

    fn offset(&self) -> u32;

    fn len(&self) -> u32;

    fn is_contiguous(&self) -> bool;

    fn binding(&self) -> Option<&(BindGroupLayout, BindGroup)>;

    fn metadata_compound(&self) -> Option<&MetadataCompound>;

    fn check_contiguous_or_view<'a>(&'a self) -> ArrayType<'a>;
}

impl ArrayCompute for GpuArray {
    fn module(&self) -> &std::sync::Arc<crate::ArrOgpuModule> {
        self.module()
    }

    fn pointer_to_arr(&self) -> [u32; 2] {
        self.pointer_to_arr()
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn stride(&self) -> &Vec<u32> {
        self.stride()
    }

    fn shape(&self) -> &Vec<u32> {
        self.shape()
    }

    fn pointer(&self) -> (u32, u32) {
        self.pointer()
    }

    fn offset(&self) -> u32 {
        0
    }

    fn len(&self) -> u32 {
        self.len() as u32
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn binding(&self) -> Option<&(BindGroupLayout, BindGroup)> {
        self.binding()
    }

    fn metadata_compound(&self) -> Option<&MetadataCompound> {
        self.metadata_compound()
    }

    fn check_contiguous_or_view<'a>(&'a self) -> ArrayType<'a> {
        self.check()
    }
}

impl<'a, A> ArrayCompute for GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    fn module(&self) -> &std::sync::Arc<crate::ArrOgpuModule> {
        self.array.module()
    }

    fn pointer_to_arr(&self) -> [u32; 2] {
        self.array.pointer_to_arr()
    }

    fn dim(&self) -> usize {
        self.shape.len()
    }

    fn stride(&self) -> &Vec<u32> {
        &self.stride
    }

    fn shape(&self) -> &Vec<u32> {
        &self.shape
    }

    fn pointer(&self) -> (u32, u32) {
        self.array.pointer()
    }

    fn offset(&self) -> u32 {
        self.offset
    }

    fn len(&self) -> u32 {
        self.shape.iter().product::<u32>()
    }

    fn is_contiguous(&self) -> bool {
        self.pointer.1 - self.pointer.0 == self.shape.iter().product::<u32>()
    }

    fn binding(&self) -> Option<&(BindGroupLayout, BindGroup)> {
        self.binding.as_ref()
    }

    fn metadata_compound(&self) -> Option<&MetadataCompound> {
        self.metadata_compound.as_ref()
    }

    fn check_contiguous_or_view(&self) -> ArrayType<'_> {
        self.check()
    }
}
