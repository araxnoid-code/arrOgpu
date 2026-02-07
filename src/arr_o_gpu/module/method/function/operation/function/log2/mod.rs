use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder, ComputePipeline, Device, PipelineLayout};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray, MetadataCompound,
    PipelineCompound,
    arr_o_gpu::compute_shaders::{LOG2_CONTIGUOUS_SHADERS_PATH, LOG2_VIEW_SHADERS_PATH},
    get_stride_from_shape, vector_padding,
};

const PIPELINE_LOG2_CONTIGUOUS: &'static str = "pipeline_log2_contiguous";
const PIPELINE_LOG2_VIEW: &'static str = "pipeline_log2_view";

impl ArrOgpuModule {
    pub fn log2<A>(&self, array: &A) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
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
            .map_err(|err| ArrOgpuErr::Log2(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Log2(err))?
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
                if let Some(pipeline) = read.get(PIPELINE_LOG2_CONTIGUOUS) {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    let pipeline = set_pipeline(
                        &wgpu.device,
                        &self.common_pipeline_layout,
                        LOG2_CONTIGUOUS_SHADERS_PATH,
                    );
                    PipelineCompound::UnsavePipeline(pipeline, PIPELINE_LOG2_CONTIGUOUS)
                }
            }
            ArrayType::View(_) => {
                if let Some(pipeline) = read.get(PIPELINE_LOG2_VIEW) {
                    PipelineCompound::PipelineCache(pipeline)
                } else {
                    let pipeline = set_pipeline(
                        &wgpu.device,
                        &self.common_pipeline_layout,
                        LOG2_VIEW_SHADERS_PATH,
                    );

                    PipelineCompound::UnsavePipeline(pipeline, PIPELINE_LOG2_VIEW)
                }
            }
        };

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Command Encoder For Log2"),
                });

        let execute_cache = &self.execute_args;
        set_execute_cache(&mut encoder, array, execute_cache, &output_metadata)?;

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Log2"),
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
        ArrayType::Contiguous(_) => PIPELINE_LOG2_CONTIGUOUS,
        ArrayType::View(_) => PIPELINE_LOG2_VIEW,
    };
    write.insert(key, pipeline);
}

fn set_execute_cache<A>(
    encoder: &mut CommandEncoder,
    array: &A,
    execute_cache: &Arc<Buffer>,
    output_metadata: &MetadataCompound,
) -> Result<(), ArrOgpuErr>
where
    A: ArrayCompute,
{
    let metadata = array.metadata_compound().ok_or(ArrOgpuErr::Log2(
        "Log2 Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    match array.check_contiguous_or_view() {
        ArrayType::Contiguous(_) => {
            encoder.copy_buffer_to_buffer(&metadata.buffer, 0, execute_cache, 0, 32);
            encoder.copy_buffer_to_buffer(&output_metadata.buffer, 0, execute_cache, 32, 32);
        }
        ArrayType::View(_) => {
            encoder.copy_buffer_to_buffer(&metadata.buffer, 0, execute_cache, 0, 256);
            encoder.copy_buffer_to_buffer(&output_metadata.buffer, 0, execute_cache, 256, 256);
        }
    }

    Ok(())
}

fn set_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    shaders_path: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Log2"),
        layout: Some(layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Log2"),
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
