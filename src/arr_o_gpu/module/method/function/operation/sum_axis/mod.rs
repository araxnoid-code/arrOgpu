use std::sync::Arc;

use bytemuck::cast_slice;
use wgpu::{Device, ShaderModule};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray,
    arr_o_gpu::{
        compute_shaders::SUM_AXIS_SHADERS_PATH,
        module::method::function::operation::sum_axis::tools::error_handling,
    },
    get_stride_from_shape, vector_padding,
};

mod sum_axis;
mod sum_axis_keep_dim;
mod tools;

impl ArrOgpuModule {
    pub fn sum_axis_unsafe<A>(&self, array_a: &A, axis: &[u32]) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let mut axis = axis.to_vec();
        axis.sort();

        error_handling(array_a, &axis)?;
        let wgpu = self.wgpu_init().read().unwrap();

        let array_shape = array_a.shape();
        let mut out_shape = array_shape.clone();
        if array_shape.len() != axis.len() {
            axis.iter().rev().for_each(|idx| {
                out_shape.remove(*idx as usize);
            });
        } else {
            out_shape = vec![1];
        }

        let dim = out_shape.len();
        let stride = get_stride_from_shape(&out_shape);
        let out_len = out_shape.iter().product::<u32>();
        let allocate = self.allocator.write().unwrap().pointer_input(out_len);

        let shape_padding: [u32; 8] = vector_padding(out_shape.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::SumAxis(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::SumAxis(err))?
            .try_into()
            .unwrap();

        let metada_output = self.create_metadata_compound(
            [allocate.1, allocate.2],
            out_len,
            dim as u32,
            0,
            shape_padding,
            stride_padding,
            stride_padding,
        );

        let pipeline = create_pipeline(&self);

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Sum Axis"),
                });

        let execute_args_buffer = &self.execute_args;

        let array_a_metadata = array_a.metadata_compound().unwrap();
        encoder.copy_buffer_to_buffer(&array_a_metadata.buffer, 0, execute_args_buffer, 0, 256);
        encoder.copy_buffer_to_buffer(&metada_output.buffer, 0, execute_args_buffer, 256, 256);

        let axis_padding =
            vector_padding(axis.clone(), 0, 8).map_err(|err| ArrOgpuErr::SumAxis(err))?;
        wgpu.queue
            .write_buffer(&self.static_cache, 0, cast_slice(&axis_padding));

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Sum Axos"),
                timestamp_writes: None,
            });

            begin_compute_pass.set_pipeline(&pipeline);
            begin_compute_pass.set_bind_group(0, Some(&self.heap_binding().binding_groups), &[]);

            let sum_len = axis
                .iter()
                .map(|axis| array_a.shape()[*axis as usize])
                .product::<u32>();

            let x = (out_len + 15) >> 4;
            let y = (sum_len + 15) >> 4;
            begin_compute_pass.dispatch_workgroups(x, y, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            binding: None,
            length: out_len as usize,
            metadata_compound: Some(metada_output),
            module: Arc::new(self.clone()),
            pointer: (allocate.1, allocate.2),
            shape: out_shape,
            space_type: allocate.0,
            stride,
        };

        Ok(array)
    }
}

fn create_pipeline(module: &ArrOgpuModule) -> wgpu::ComputePipeline {
    let device = &module.wgpu_init().read().unwrap().device;

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Create Pipeline Layout For Sum Axis"),
        bind_group_layouts: &[&module.heap_binding().binding_group_layouts],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Sum Axis"),
        layout: Some(&pipeline_layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Sum Axis"),
            source: wgpu::ShaderSource::Wgsl(SUM_AXIS_SHADERS_PATH.into()),
        }),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    pipeline
}
