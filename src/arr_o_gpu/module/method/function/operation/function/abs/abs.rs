use std::sync::Arc;

use wgpu::{Device, ShaderModule};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, CheckArrayType, GpuArray, get_stride_from_shape,
};

impl ArrOgpuModule {
    pub fn abs<'a, A>(&self, array: &'a A) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
    {
        // output metada
        let len = array.len();
        let allocate = self.allocator.write().unwrap().pointer_input(len);
        let shape = array.shape().clone();
        let stride = get_stride_from_shape(&shape);

        // bind
        let heap_bind = self.heap_binding();
        let array_bind = array.binding();
        let out_bind =
            self.create_metadata_binding(&[allocate.1, allocate.2], &shape, &stride, &stride, &0);

        let wgpu = self.wgpu_init().read().unwrap();
        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Abs"),
                bind_group_layouts: &[&heap_bind.binding_group_layouts, &array_bind.0, &out_bind.0],
                push_constant_ranges: &[],
            });

        let pipeline = wgpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Abs"),
                layout: Some(&pipeline_layout),
                module: &get_shaders::<A>(&wgpu.device, &array),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Abs"),
                });

        {
            let mut bcp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Abs"),
                timestamp_writes: None,
            });

            bcp.set_pipeline(&pipeline);
            bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            bcp.set_bind_group(1, Some(&array_bind.1), &[]);
            bcp.set_bind_group(2, Some(&out_bind.1), &[]);

            let x = (len + 255) / 256;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            module: Arc::new(self.clone()),
            binding: out_bind,
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            shape,
            stride,
            space_type: allocate.0,
        };

        Ok(array)
    }
}

fn get_shaders<'a, A>(device: &Device, arr: &'a A) -> ShaderModule
where
    A: CheckArrayType<'a>,
{
    match arr.check() {
        crate::ArrayType::Contiguous(_) => {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Create Shaders Module"),
                source: wgpu::ShaderSource::Wgsl(include_str!("./wgsl/abs_contiguous.wgsl").into()),
            })
        }
        crate::ArrayType::View(_) => device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./wgsl/abs_view.wgsl").into()),
        }),
    }
}
