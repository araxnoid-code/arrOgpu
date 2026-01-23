use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device, PipelineLayout, ShaderModule,
    util::DeviceExt,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray, PipelineCompound,
    arr_o_gpu::{
        compute_shaders::{SUB_SCALAR_CONTIGUOUS_SHADERS_PATH, SUB_SCALAR_VIEW_SHADERS_PATH},
        module::method::function::operation::element_wise::skalar_operation::MetaDataOption,
    },
    get_stride_from_shape, vector_padding,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct StaticInterface {
    counter: u32,
    scalar: f32,
    index: u32,
    padding: u32,
}

impl ArrOgpuModule {
    pub(crate) fn sub_skalar<A>(
        &self,
        array: &A,
        meta_data_option: MetaDataOption,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
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

        let read = self.pipeline_cache.read().unwrap();
        let pipeline = match array.check_contiguous_or_view() {
            ArrayType::Contiguous(_) => {
                if let Some(pipeline) = read.get("pipeline_sub_scalar_contiguous") {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    drop(read);
                    let pipeline_layout = &self.common_pipeline_layout;
                    PipelineCompound::UnsavePipeline(
                        set_pipeline(&wgpu.device, &pipeline_layout, array),
                        "pipeline_sub_scalar_contiguous",
                    )
                }
            }
            ArrayType::View(_) => {
                if let Some(pipeline) = read.get("pipeline_sub_scalar_view") {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    drop(read);
                    let pipeline_layout = &self.common_pipeline_layout;
                    PipelineCompound::UnsavePipeline(
                        set_pipeline(&wgpu.device, &pipeline_layout, array),
                        "pipeline_sub_scalar_view",
                    )
                }
            }
        };

        let mut encoder = wgpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Create Encoder For Sub"),
            });

        set_cache(
            self,
            &wgpu.device,
            &mut encoder,
            array,
            &output_metadata.buffer,
            &meta_data_option,
        )?;

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Sub"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);
            begin_compute_pass.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            let x = (len + 255) >> 8;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }
        wgpu.queue.submit(Some(encoder.finish()));

        if let PipelineCompound::UnsavePipeline(pipeline, _) = pipeline {
            set_pipeline_cache(self, array, pipeline);
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

fn set_pipeline_cache<A>(module: &ArrOgpuModule, array: &A, pipeline: ComputePipeline)
where
    A: ArrayCompute,
{
    let mut write = module.pipeline_cache.write().unwrap();
    let key = match array.check_contiguous_or_view() {
        ArrayType::Contiguous(_) => "pipeline_sub_scalar_contiguous",
        ArrayType::View(_) => "pipeline_sub_scalar_view",
    };

    write.insert(key, pipeline);
}

fn set_cache<A>(
    module: &ArrOgpuModule,
    device: &Device,
    encoder: &mut CommandEncoder,
    array: &A,
    out_metadata_buffer: &Buffer,
    option: &MetaDataOption,
) -> Result<(), ArrOgpuErr>
where
    A: ArrayCompute,
{
    let execute_args = &module.execute_args;
    let static_cache = &module.static_cache;

    let array_a_metadata_buffer = array.metadata_compound().ok_or(ArrOgpuErr::Sub(
        "Sub Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    let size = if let ArrayType::Contiguous(_) = array.check_contiguous_or_view() {
        32
    } else {
        256
    };

    encoder.copy_buffer_to_buffer(&array_a_metadata_buffer.buffer, 0, &execute_args, 0, size);
    encoder.copy_buffer_to_buffer(out_metadata_buffer, 0, &execute_args, size, size);

    let (counter, scalar, index) = match option {
        MetaDataOption::Array(arr) => {
            let scalar_index = arr.pointer().0 + arr.offset();
            (1, 0., scalar_index)
        }
        MetaDataOption::Skalar(scalar) => (0, *scalar, 0),
    };

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Create Static Interface Buffer For Sub"),
        contents: bytemuck::bytes_of(&StaticInterface {
            counter,
            scalar,
            index,
            padding: 0,
        }),
        usage: BufferUsages::COPY_SRC,
    });

    encoder.copy_buffer_to_buffer(&buffer, 0, &static_cache, 0, 16);

    Ok(())
}

fn set_pipeline<A>(device: &Device, pipeline_layout: &PipelineLayout, array: &A) -> ComputePipeline
where
    A: ArrayCompute,
{
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline Layout For Sub"),
        layout: Some(&pipeline_layout),
        module: &set_shaders(&device, array),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
}

fn set_shaders<A>(device: &Device, array: &A) -> ShaderModule
where
    A: ArrayCompute,
{
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Create Shaders Module For Sub"),
        source: wgpu::ShaderSource::Wgsl(match array.check_contiguous_or_view() {
            ArrayType::Contiguous(_) => SUB_SCALAR_CONTIGUOUS_SHADERS_PATH.into(),
            ArrayType::View(_) => SUB_SCALAR_VIEW_SHADERS_PATH.into(),
        }),
    })
}
