use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder, Device, ShaderModule};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, CheckArrayType, GpuArray,
    get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub(crate) fn add_array_metadata<'a, A, B>(
        &self,
        array_a: &'a A,
        array_b: &'a B,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: ArrayCompute + ?Sized,
    {
        let wgpu = self.wgpu_init.read().unwrap();
        // metadata output
        let len = array_a.len();
        let dim = array_a.dim();
        let offset = 0;
        let shape = array_a.shape().clone();
        let stride = get_stride_from_shape(&shape);
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        let shape_padding: [u32; 12] = vector_padding(shape.clone(), 0, 12)?.try_into().unwrap();
        let origin_stride_padding: [u32; 12] =
            vector_padding(stride.clone(), 0, 12)?.try_into().unwrap();
        let output_metadata = self.create_metadata_compound(
            [allocate.1, allocate.2],
            len,
            dim as u32,
            offset,
            shape_padding,
            origin_stride_padding,
            origin_stride_padding,
        );

        // bind group
        // // heap
        let heap_bind = &self.heap_binding();

        // pipeline
        let shaders = set_shaders(&wgpu.device, array_a, array_b);

        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Create Pipeline For Add"),
                bind_group_layouts: &[&heap_bind.binding_group_layouts],
                immediate_size: 0,
            });

        let pipeline = wgpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Create Pipeline For Add"),
                layout: Some(&pipeline_layout),
                module: &shaders,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &*get_override(array_a, array_b, allocate.1 as f64),
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Add"),
                });

        set_execute_args(&mut encoder, &self.execute_args, array_a, array_b)?;

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass"),
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

fn set_shaders<'a, A, B>(device: &Device, array_a: &'a A, array_b: &'a B) -> ShaderModule
where
    A: ArrayCompute,
    B: ArrayCompute + ?Sized,
{
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Create Shaders Module For Add"),
        source: wgpu::ShaderSource::Wgsl(
            match (
                array_a.check_contiguous_or_view(),
                array_b.check_contiguous_or_view(),
            ) {
                (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => {
                    include_str!("./shaders/add_contiguous_metadata.wgsl").into()
                }
                _ => include_str!("./shaders/add_view_metadata.wgsl").into(),
            },
        ),
    })
}

fn set_execute_args<'a, A, B>(
    encoder: &mut CommandEncoder,
    execute_cache: &Arc<Buffer>,
    array_a: &'a A,
    array_b: &'a B,
) -> Result<(), ArrOgpuErr>
where
    A: ArrayCompute,
    B: ArrayCompute + ?Sized,
{
    let buffer_a = array_a.metadata_compound().ok_or(ArrOgpuErr::Add(
        "Add Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    let buffer_b = array_b.metadata_compound().ok_or(ArrOgpuErr::Add(
        "Add Error, Metadata Not Yet Defined For Array B".to_string(),
    ))?;

    match (
        array_a.check_contiguous_or_view(),
        array_b.check_contiguous_or_view(),
    ) {
        (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => (),
        _ => {
            encoder.copy_buffer_to_buffer(&buffer_a.buffer, 0, execute_cache, 0, 256);
            encoder.copy_buffer_to_buffer(&buffer_b.buffer, 0, execute_cache, 256, 256);
        }
    }

    Ok(())
}

pub fn get_override<'a, A, B>(
    array_a: &'a A,
    array_b: &'a B,
    start_pointer_o: f64,
) -> Box<[(&'a str, f64)]>
where
    A: ArrayCompute,
    B: ArrayCompute + ?Sized,
{
    match (
        array_a.check_contiguous_or_view(),
        array_b.check_contiguous_or_view(),
    ) {
        (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => Box::new([
            ("LEN", array_a.len() as f64),
            ("START_POINTER_A", array_a.pointer().0 as f64),
            ("START_POINTER_B", array_b.pointer().0 as f64),
            ("START_POINTER_O", start_pointer_o),
        ]),

        _ => Box::new([
            ("LEN", array_a.len() as f64),
            ("DIM", array_a.dim() as f64),
            ("START_POINTER_A", array_a.pointer().0 as f64),
            ("START_POINTER_B", array_b.pointer().0 as f64),
            ("START_POINTER_O", start_pointer_o),
            ("OFFSET_A", array_a.offset() as f64),
            ("OFFSET_B", array_b.offset() as f64),
        ]),
    }
}
