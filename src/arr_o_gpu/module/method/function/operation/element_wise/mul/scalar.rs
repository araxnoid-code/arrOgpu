use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device, PipelineLayout, ShaderModule,
    util::DeviceExt,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray, PipelineCompound,
    arr_o_gpu::{
        compute_shaders::{MUL_SCALAR_CONTIGUOUS_SHADERS_PATH, MUL_SCALAR_VIEW_SHADERS_PATH},
        module::method::function::operation::element_wise::skalar_operation::MetaDataOption,
    },
    get_stride_from_shape, vector_padding,
};

const PIPELINE_SCALAR_MUL_CONTIGUOUS: &'static str = "pipeline_mul_scalar_contiguous";
const PIPELINE_SCALAR_MUL_VIEW: &'static str = "pipeline_mul_scalar_view";

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct StaticInterface {
    counter: u32,
    scalar: f32,
    index: u32,
    padding: u32,
}

impl ArrOgpuModule {
    pub(crate) fn mul_skalar<A>(
        &self,
        array: &A,
        meta_data_option: MetaDataOption,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let wgpu = self.wgpu_module.read().unwrap();
        // output metadata
        let len = array.len();
        let dim = array.dim();
        let offset = 0;
        let shape = array.shape().clone();
        let stride = get_stride_from_shape(&shape);
        let allocated = self
            .allocator
            .write()
            .unwrap()
            .allocate(len)
            .map_err(|msg| ArrOgpuErr::Allocate(msg))?;
        let pointer = allocated.get_range();

        let shape_padding: [u32; 8] = vector_padding(shape.clone(), 0, 8)?.try_into().unwrap();
        let origin_stride_padding: [u32; 8] =
            vector_padding(stride.clone(), 0, 8)?.try_into().unwrap();
        let output_metadata = self.create_metadata_compound(
            [pointer.0 as u32, pointer.1 as u32],
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
                if let Some(pipeline) = read.get(PIPELINE_SCALAR_MUL_CONTIGUOUS) {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    drop(read);
                    let pipeline_layout = &self.common_pipeline_layout;
                    PipelineCompound::UnsavePipeline(
                        create_pipeline(
                            &wgpu.device,
                            &pipeline_layout,
                            array,
                            self.function_execute_opt.mul.compute_workgroup_size_x,
                        ),
                        PIPELINE_SCALAR_MUL_CONTIGUOUS,
                    )
                }
            }
            ArrayType::View(_) => {
                if let Some(pipeline) = read.get(PIPELINE_SCALAR_MUL_VIEW) {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    drop(read);
                    let pipeline_layout = &self.common_pipeline_layout;
                    PipelineCompound::UnsavePipeline(
                        create_pipeline(
                            &wgpu.device,
                            &pipeline_layout,
                            array,
                            self.function_execute_opt.mul.compute_workgroup_size_x,
                        ),
                        PIPELINE_SCALAR_MUL_VIEW,
                    )
                }
            }
        };

        let mut encoder = wgpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Create Encoder For Mul"),
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
                label: Some("Create Begin Compute Pass For Mul"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);
            begin_compute_pass.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);

            let workgroup_size_x = self.function_execute_opt.mul.compute_workgroup_size_x;
            let x = (len + workgroup_size_x - 1) / workgroup_size_x;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }
        wgpu.queue.submit(Some(encoder.finish()));

        self.saving_from_pipeline_compound(pipeline.get_unsave_pipeline());

        let array = GpuArray {
            module: Arc::new(self.clone()),
            length: len as usize,
            metadata_compound: Some(output_metadata),
            shape,
            stride,
            allocated,
        };

        Ok(array)
    }
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

    let array_a_metadata_buffer = array.metadata_compound().ok_or(ArrOgpuErr::Mul(
        "Mul Error, Metadata Not Yet Defined For Array A".to_string(),
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
        label: Some("Create Static Interface Buffer For Mul"),
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

fn create_pipeline<A>(
    device: &Device,
    pipeline_layout: &PipelineLayout,
    array: &A,
    workgroup_size_x: u32,
) -> ComputePipeline
where
    A: ArrayCompute,
{
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline Layout For Mul"),
        layout: Some(&pipeline_layout),
        module: &create_shaders(&device, array),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("WORKGROUP_SIZE_X", workgroup_size_x as f64)],
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
}

fn create_shaders<A>(device: &Device, array: &A) -> ShaderModule
where
    A: ArrayCompute,
{
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Create Shaders Module For Mul"),
        source: wgpu::ShaderSource::Wgsl(match array.check_contiguous_or_view() {
            ArrayType::Contiguous(_) => MUL_SCALAR_CONTIGUOUS_SHADERS_PATH.into(),
            ArrayType::View(_) => MUL_SCALAR_VIEW_SHADERS_PATH.into(),
        }),
    })
}
