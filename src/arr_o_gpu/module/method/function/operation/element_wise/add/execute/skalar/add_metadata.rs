use std::sync::Arc;

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::MetaDataOption,
    get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub(crate) fn add_skalar_metadata<A>(
        &self,
        array: &A,
        meta_data_option: MetaDataOption,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        if let ArrayType::View(_) = array.check_contiguous_or_view() {
            let error = "Add ERROR, View Array Not Supported Yet For Scalar Operation".to_string();
            return Err(ArrOgpuErr::Add(error));
        }

        let wgpu = self.wgpu_init.read().unwrap();
        // output metadata
        let len = array.len();
        let dim = array.dim();
        let offset = 0;
        let shape = array.shape().clone();
        let stride = get_stride_from_shape(&shape);
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        let shape_padding: [u32; 8] = vector_padding(shape.clone(), 0, 8)?.try_into().unwrap();
        let origin_stride_padding: [u32; 8] =
            vector_padding(stride.clone(), 0, 8)?.try_into().unwrap();
        let output_metadata = self.create_metadata_compound(
            [allocate.1, allocate.2],
            len,
            dim as u32,
            offset,
            shape_padding,
            origin_stride_padding,
            origin_stride_padding,
        );

        // heap
        let heap_bind = self.heap_binding();

        let shaders = wgpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Create Shaders Module For Add"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./shaders/add_skalar_contiguous_metadata.wgsl").into(),
                ),
            });

        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Add"),
                bind_group_layouts: &[&heap_bind.binding_group_layouts],
                immediate_size: 0,
            });

        let pipeline = wgpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Add"),
                layout: Some(&pipeline_layout),
                module: &shaders,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &get_override(array, &meta_data_option, allocate.1 as f64),
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

        let mut encoder = wgpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Create Encoder For Add"),
            });

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Add"),
                timestamp_writes: None,
            });

            begin_compute_pass.set_pipeline(&pipeline);
            begin_compute_pass.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            let x = (len + 255) >> 8;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            module: Arc::new(self.clone()),
            length: len as usize,
            binding: None,
            metadata_compound: Some(output_metadata),
            pointer: (allocate.1, allocate.2),
            shape,
            space_type: allocate.0,
            stride,
        };

        Ok(array)
    }
}

fn get_override<'a, A>(
    array: &A,
    meta_data_option: &MetaDataOption,
    start_pointer_o: f64,
) -> [(&'a str, f64); 6]
where
    A: ArrayCompute,
{
    let (scalar_index, scalar, scalar_counter) = match meta_data_option {
        MetaDataOption::Array(arr) => ((arr.offset() + arr.pointer().0) as f64, 0., 1.),
        MetaDataOption::Skalar(scalar) => (0., *scalar as f64, 0.),
    };

    [
        ("LEN", array.len() as f64),
        ("START_POINTER", array.pointer().0 as f64),
        ("START_POINTER_O", start_pointer_o),
        ("SCALAR_INDEX", scalar_index),
        ("SCALAR", scalar),
        ("SCALAR_COUNTER", scalar_counter),
    ]
}
