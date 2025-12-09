use std::sync::Arc;

use wgpu::{
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    wgt::PollType,
};

use crate::{
    ArrOgpuErr,
    ArrOgpuModule,
    ArrayView,
    GpuArray,
    GpuArrayView,
    SliceRange,
    bind_group_slicing,
    get_stride_from_shape,
};

impl ArrOgpuModule {
    pub fn slicing(&self, array: &GpuArray, slice: &[SliceRange]) -> Result<GpuArray, ArrOgpuErr> {
        if array.shape().len() < slice.len() || slice.len() == 0 {
            let err = format!(
                "Array Slicing Error, Array {:?} can't Slice By {:?} cause out of range",
                array.shape(),
                slice
            );

            return Err(ArrOgpuErr::Slicing(err));
        }

        // check & get
        let wgpu_init = self.wgpu_init.read().unwrap();
        let mut allocator = self.allocator.write().unwrap();

        let mut output_shape = vec![];
        let mut start_slice = vec![];
        let mut end_slice = vec![];
        for (i, _) in array.shape().iter().enumerate() {
            if let Some(range) = slice.get(i) {
                let start = range.start.unwrap_or(0);
                let end = range.end.unwrap_or(array.shape()[i]);

                if start >= end || end > array.shape()[i] {
                    let err = format!(
                        "Array Slicing Error, Error detected for {:?} in slice {:?}",
                        range,
                        slice
                    );
                    return Err(ArrOgpuErr::Slicing(err));
                }

                start_slice.push(start);
                end_slice.push(end);
                output_shape.push((end - start) as u32);
            } else {
                output_shape.push(array.shape()[i] as u32);
            }
        }

        let output_allocate = allocator.pointer_input(output_shape.iter().product());
        let output_pointer = [output_allocate.1, output_allocate.2];

        let array_stride = array.stride();
        let array_pointer = array.pointer_to_arr();

        let total_unit = output_shape[..slice.len() - 1].iter().product::<u32>();

        let output_stride = get_stride_from_shape(&output_shape);
        let mut _loop = vec![];
        for i in 0..&output_shape[..slice.len()].len() - 1 {
            let iter = output_shape[..slice.len()][i + 1..output_shape[..slice.len()].len() - 1]
                .iter()
                .product::<u32>();
            _loop.push(iter);
        }
        if _loop.len() == 0 {
            _loop = vec![1];
        }

        let (bind_group_layout, bind_group) = bind_group_slicing(
            &wgpu_init,
            &start_slice,
            &end_slice,
            &_loop,
            &array_stride,
            &array_pointer,
            &output_pointer,
            total_unit
        );

        // pipeline
        let bind_group_heap = &self.binding_compounds.read().unwrap()[0];
        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Slicing"),
                bind_group_layouts: &[
                    // heap
                    &bind_group_heap.binding_group_layouts,
                    //slicing
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            })
        );

        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader For Slicing"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./slicing.wgsl").into()),
        });

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Slicing"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        // encoder
        let mut encoder = wgpu_init.device.create_command_encoder(
            &(wgpu::wgt::CommandEncoderDescriptor {
                label: Some("Create Encoder For Slicing"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(wgpu::ComputePassDescriptor {
                    label: Some("Create COmpute Pass Descriptor For Slicing"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);
            // group 0
            bcp.set_bind_group(0, Some(&bind_group_heap.binding_groups), &[]);
            // group 1
            bcp.set_bind_group(1, Some(&bind_group), &[]);

            let last_start = start_slice.last().unwrap();
            let last_end = end_slice.last().unwrap();
            let start = last_start * array_stride[slice.len() - 1];
            let end = last_end * array_stride[slice.len() - 1];
            let len = end - start;

            let x = ((total_unit as f32) / 16.0).ceil() as u32;
            let y = ((len as f32) / 16.0).ceil() as u32;
            bcp.dispatch_workgroups(x, y, 1);
        }
        wgpu_init.queue.submit(Some(encoder.finish()));

        wgpu_init.device.poll(PollType::Wait).unwrap();

        let arr = GpuArray {
            length: output_shape.iter().product::<u32>() as usize,
            module: Arc::new(self.clone()),
            pointer: (output_allocate.1, output_allocate.2),
            stride: output_stride,
            shape: output_shape,
            space_type: output_allocate.0,
        };

        Ok(arr)
    }
}
