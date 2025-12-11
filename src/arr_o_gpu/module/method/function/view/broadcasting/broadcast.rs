use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    wgt::{ CommandEncoderDescriptor, PollType },
};

use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray, broadcast_bind_group, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn broadcasting(&self, arr: &GpuArray, broadcast: &[u32]) -> Result<GpuArray, ArrOgpuErr> {
        let arr_shape = arr.shape();
        if arr_shape.len() > broadcast.len() {
            let err = format!(
                "Array Broadcasting Error, Array {:?} can't Broadcast to {:?}",
                arr_shape,
                broadcast
            );

            return Err(ArrOgpuErr::Broadcast(err));
        }

        // expend shape
        let diff_range = broadcast.len() - arr_shape.len();
        let extend_arr_shape = if diff_range != 0 {
            let mut extend = vec![1;diff_range];
            extend.extend_from_slice(arr_shape);
            extend
        } else {
            arr_shape.clone()
        };

        // validation every dim
        let mut broadcast_index = None;
        for i in (0..broadcast.len()).rev() {
            let arr_dim = extend_arr_shape[i];
            let broadcast_dim = broadcast[i];

            if arr_dim != broadcast_dim {
                if arr_dim == 1 {
                    if let None = broadcast_index {
                        broadcast_index = Some(i);
                    } else {
                        let err = format!(
                            "Array Broadcasting Error, Array {:?} can't Broadcast to {:?} Cause Broadcast can't over then one target",
                            arr_shape,
                            broadcast
                        );

                        return Err(ArrOgpuErr::Broadcast(err));
                    }
                } else {
                    let err = format!(
                        "Array Broadcasting Error, Array {:?} can't Broadcast to {:?}",
                        arr_shape,
                        broadcast
                    );

                    return Err(ArrOgpuErr::Broadcast(err));
                }
            }
        }

        let wgpu_init = self.wgpu_init.read().unwrap();
        let allocator = self.allocator_write();
        // binding
        let heap_binding = &self.binding_compounds.read().unwrap()[0];
        let (bind_group_layout, bind_group, thread_limit, stride_out, output_allocate) =
            broadcast_bind_group(
                &wgpu_init.device,
                arr,
                allocator,
                broadcast,
                broadcast_index.unwrap()
            );

        // pipeline
        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Broadcast"),
                bind_group_layouts: &[&heap_binding.binding_group_layouts, &bind_group_layout],
                push_constant_ranges: &[],
            })
        );

        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Broadcast"),
            source: ShaderSource::Wgsl(include_str!("./broadcast.wgsl").into()),
        });

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("()"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Broadcast"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Broadcast"),
                    timestamp_writes: None,
                })
            );

            // pipeline
            bcp.set_pipeline(&pipeline);

            // group
            // // 0
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);
            // // 1
            bcp.set_bind_group(1, Some(&bind_group), &[]);

            // dispatch workgroup
            let x = ((thread_limit as f32) / 16.0).ceil() as u32;
            let y = ((stride_out as f32) / 16.0).ceil() as u32;
            bcp.dispatch_workgroups(x, y, 1);
        }
        wgpu_init.queue.submit(Some(encoder.finish()));
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let out_shape = broadcast.to_vec();
        let stride = get_stride_from_shape(&out_shape);

        let binding = self.array_data_binding(
            &[output_allocate.1, output_allocate.2],
            &out_shape,
            &stride,
            &stride,
            &0
        );

        let len = out_shape.iter().product::<u32>() as usize;
        let arr = GpuArray {
            length: len,
            module: Arc::new(self.clone()),
            shape: out_shape,
            stride,
            pointer: (output_allocate.1, output_allocate.2),
            space_type: output_allocate.0,
            binding: binding,
        };
        Ok(arr)
    }
}
