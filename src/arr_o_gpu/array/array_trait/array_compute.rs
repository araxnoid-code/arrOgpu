use std::process::Output;

use wgpu::{BindGroup, BindGroupLayout, naga::Type};

use crate::{GpuArray, GpuArrayView};

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

    fn binding(&self) -> &(BindGroupLayout, BindGroup);
}

impl ArrayCompute for GpuArray {
    fn module(&self) -> &std::sync::Arc<crate::ArrOgpuModule> {
        &self.module
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
        self.pointer
    }

    fn offset(&self) -> u32 {
        0
    }

    fn len(&self) -> u32 {
        self.length as u32
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn binding(&self) -> &(BindGroupLayout, BindGroup) {
        &self.binding
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

    fn binding(&self) -> &(BindGroupLayout, BindGroup) {
        &self.binding
    }

    // fn get_array_type<B>(&self) -> ArrayType<'_, A> {
    //     ArrayType::ViewArray(self)
    // }
}
