use wgpu::{ BufferUsages, util::{ BufferInitDescriptor, DeviceExt } };

use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray, get_stride_from_shape, matmul_nd_group_binding };

impl ArrOgpuModule {
    pub fn matmul_nd(&self, arr_a: &GpuArray, arr_b: &GpuArray) -> Result<(), ArrOgpuErr> {
        let shape_a = arr_a.shape();
        let k_a = shape_a[shape_a.len() - 1];
        let shape_b = arr_b.shape();
        let k_b = shape_b[shape_b.len() - 2];

        if
            shape_a.len() != shape_b.len() ||
            &shape_a[0..shape_a.len() - 2] != &shape_b[0..shape_a.len() - 2] ||
            k_a != k_b
        {
            let err = format!(
                "Array matmul nd Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a,
                shape_b
            );

            return Err(ArrOgpuErr::MatmulND(err));
        }

        let wgpu_init = self.wgpu_init.read().unwrap();
        matmul_nd_group_binding(&wgpu_init.device, &self.allocator, arr_a, arr_b);

        // others

        Ok(())
    }
}
