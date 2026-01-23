use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder, ComputePipeline, Device, PipelineLayout, ShaderModule};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, CheckArrayType, GpuArray, PipelineCompound,
    arr_o_gpu::compute_shaders::{
        ADD_NON_SCALAR_CONTIGUOUS_SHADERS_PATH, ADD_NON_SCALAR_VIEW_SHADERS_PATH,
    },
    get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub(crate) fn add_array<'a, A, B>(
        &self,
        array_a: &'a A,
        array_b: &'a B,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: ArrayCompute + ?Sized,
    {
        if array_a.shape() != array_b.shape() {
            let err = format!(
                "Array Add Error, Shape Of A is {:?} but adding with Shape Of B is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            return Err(ArrOgpuErr::Add(err));
        }

        let wgpu = self.wgpu_init.read().unwrap();
        // metadata output
        let len = array_a.len();
        let dim = array_a.dim();
        let offset = 0;
        let shape = array_a.shape().clone();
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

        // bind group
        // // heap
        let heap_bind = &self.heap_binding();

        // pipeline
        let read = self.pipeline_cache.read().unwrap();
        let pipeline = match (
            array_a.check_contiguous_or_view(),
            array_b.check_contiguous_or_view(),
        ) {
            (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => {
                if let Some(pipeline) = read.get("pipeline_add_non_scalar_contiguous") {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    drop(read);
                    PipelineCompound::UnsavePipeline(
                        set_pipeline(&wgpu.device, array_a, array_b, &self.common_pipeline_layout),
                        "pipeline_add_non_scalar_contiguous",
                    )
                }
            }

            _ => {
                if let Some(pipeline) = read.get("pipeline_add_non_scalar_view") {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    drop(read);
                    PipelineCompound::UnsavePipeline(
                        set_pipeline(&wgpu.device, array_a, array_b, &self.common_pipeline_layout),
                        "pipeline_add_non_scalar_view",
                    )
                }
            }
        };

        let mut encoder = wgpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Create Encoder For Add"),
            });

        set_execute_args(
            &mut encoder,
            &self.execute_args,
            array_a,
            array_b,
            &output_metadata.buffer,
        )?;

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);
            begin_compute_pass.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            let x = (len + 255) >> 8;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        if let PipelineCompound::UnsavePipeline(pipeline, _) = pipeline {
            save_pipeline(self, pipeline, array_a, array_b);
        }

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

fn save_pipeline<'a, A, B>(
    module: &ArrOgpuModule,
    pipeline: ComputePipeline,
    array_a: &'a A,
    array_b: &'a B,
) where
    A: ArrayCompute,
    B: ArrayCompute + ?Sized,
{
    let mut write = module.pipeline_cache.write().unwrap();
    match (
        array_a.check_contiguous_or_view(),
        array_b.check_contiguous_or_view(),
    ) {
        (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => {
            write.insert("pipeline_add_non_scalar_contiguous", pipeline);
        }
        _ => {
            write.insert("pipeline_add_non_scalar_view", pipeline);
        }
    };
}

fn set_pipeline<'a, A, B>(
    device: &Device,
    array_a: &'a A,
    array_b: &'a B,
    pipeline_layout: &PipelineLayout,
) -> ComputePipeline
where
    A: ArrayCompute,
    B: ArrayCompute + ?Sized,
{
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Add"),
        layout: Some(&pipeline_layout),
        module: &set_shaders(&device, array_a, array_b),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
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
                    ADD_NON_SCALAR_CONTIGUOUS_SHADERS_PATH.into()
                }
                _ => ADD_NON_SCALAR_VIEW_SHADERS_PATH.into(),
            },
        ),
    })
}

fn set_execute_args<'a, A, B>(
    encoder: &mut CommandEncoder,
    execute_cache: &Arc<Buffer>,
    array_a: &'a A,
    array_b: &'a B,
    metadata_o_buffer: &Buffer,
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
        (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => {
            encoder.copy_buffer_to_buffer(&buffer_a.buffer, 0, execute_cache, 0, 32);
            encoder.copy_buffer_to_buffer(&buffer_b.buffer, 0, execute_cache, 32, 32);
            encoder.copy_buffer_to_buffer(&metadata_o_buffer, 0, execute_cache, 64, 32);
        }
        _ => {
            encoder.copy_buffer_to_buffer(&buffer_a.buffer, 0, execute_cache, 0, 256);
            encoder.copy_buffer_to_buffer(&buffer_b.buffer, 0, execute_cache, 256, 256);
            encoder.copy_buffer_to_buffer(&metadata_o_buffer, 0, execute_cache, 512, 256);
        }
    }

    Ok(())
}
