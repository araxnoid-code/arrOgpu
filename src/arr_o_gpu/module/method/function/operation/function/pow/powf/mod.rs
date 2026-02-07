use std::sync::Arc;

use wgpu::{
    Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device, PipelineLayout, util::DeviceExt,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray, MetadataCompound,
    PipelineCompound,
    arr_o_gpu::{
        compute_shaders::{POWF_CONTIGUOUS_SHADERS_PATH, POWF_VIEW_SHADERS_PATH},
        module::method::function::operation::function::pow::{PowFloat, PowTrait},
    },
    get_stride_from_shape, vector_padding,
};

const PIPELINE_POWF_CONTIGUOUS: &'static str = "pipeline_powf_contiguous";
const PIPELINE_POWF_VIEW: &'static str = "pipeline_powf_view";

impl ArrOgpuModule {
    pub fn powf<A, B>(&self, array: &A, power: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
        B: PowTrait + PowFloat,
    {
        let wgpu = self.wgpu_init().read().unwrap();

        // output metadata
        let len = array.len();
        let dim = array.dim();
        let shape = array.shape();
        let stride = get_stride_from_shape(&shape);
        let offset = 0;
        let allocated = self
            .allocator
            .write()
            .unwrap()
            .allocate(len)
            .map_err(|msg| ArrOgpuErr::Allocate(msg))?;
        let pointer = allocated.get_range();

        // padding
        let shape_padding: [u32; 8] = vector_padding(shape.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Powf(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Powf(err))?
            .try_into()
            .unwrap();

        let output_metadata = self.create_metadata_compound(
            [pointer.0 as u32, pointer.1 as u32],
            len,
            dim as u32,
            offset,
            shape_padding,
            stride_padding,
            stride_padding,
        );

        let read = self.pipeline_cache.read().unwrap();
        let pipeline = match array.check_contiguous_or_view() {
            ArrayType::Contiguous(_) => {
                if let Some(pipeline) = read.get(PIPELINE_POWF_CONTIGUOUS) {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    let pipeline = set_pipeline(
                        &wgpu.device,
                        &self.common_pipeline_layout,
                        POWF_CONTIGUOUS_SHADERS_PATH,
                    );
                    PipelineCompound::UnsavePipeline(pipeline, PIPELINE_POWF_CONTIGUOUS)
                }
            }
            ArrayType::View(_) => {
                if let Some(pipeline) = read.get(PIPELINE_POWF_VIEW) {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    let pipeline = set_pipeline(
                        &wgpu.device,
                        &self.common_pipeline_layout,
                        POWF_VIEW_SHADERS_PATH,
                    );

                    PipelineCompound::UnsavePipeline(pipeline, PIPELINE_POWF_VIEW)
                }
            }
        };

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Command Encoder For Powf"),
                });

        let execute_args = &self.execute_args;
        let static_cache = &self.static_cache;

        set_cache(
            &wgpu.device,
            &mut encoder,
            array,
            power,
            execute_args,
            static_cache,
            &output_metadata,
        )?;

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Powf"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);
            begin_compute_pass.set_bind_group(0, Some(&self.heap_binding().binding_groups), &[]);
            let x = (len + 255) / 256;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        if let PipelineCompound::UnsavePipeline(pipeline, _) = pipeline {
            drop(read);
            set_pipeline_cache(self, array, pipeline);
        }

        let array = GpuArray {
            module: Arc::new(self.clone()),
            length: len as usize,
            shape: shape.clone(),
            stride: stride,
            metadata_compound: Some(output_metadata),
            allocated,
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
        ArrayType::Contiguous(_) => PIPELINE_POWF_CONTIGUOUS,
        ArrayType::View(_) => PIPELINE_POWF_VIEW,
    };
    write.insert(key, pipeline);
}

fn set_cache<A, B>(
    device: &Device,
    encoder: &mut CommandEncoder,
    array: &A,
    powf: &B,
    execute_args: &Arc<Buffer>,
    static_cache: &Arc<Buffer>,
    output_metadata: &MetadataCompound,
) -> Result<(), ArrOgpuErr>
where
    A: ArrayCompute,
    B: PowTrait,
{
    let metadata = array.metadata_compound().ok_or(ArrOgpuErr::Powf(
        "Powf Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    match array.check_contiguous_or_view() {
        ArrayType::Contiguous(_) => {
            encoder.copy_buffer_to_buffer(&metadata.buffer, 0, execute_args, 0, 32);
            encoder.copy_buffer_to_buffer(&output_metadata.buffer, 0, execute_args, 32, 32);
        }
        ArrayType::View(_) => {
            encoder.copy_buffer_to_buffer(&metadata.buffer, 0, execute_args, 0, 256);
            encoder.copy_buffer_to_buffer(&output_metadata.buffer, 0, execute_args, 256, 256);
        }
    }

    let static_data = powf
        .get()
        .map_err(|msg| ArrOgpuErr::Log2(msg.to_string()))?;
    let static_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Create Static Buffer For Powf"),
        contents: bytemuck::bytes_of(&static_data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&static_buffer, 0, &static_cache, 0, 16);

    Ok(())
}

fn set_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    shaders_path: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Powf"),
        layout: Some(layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Powf"),
            source: wgpu::ShaderSource::Wgsl(shaders_path.into()),
        }),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
}
