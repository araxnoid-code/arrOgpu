mod matmul_operate;
use std::sync::Arc;
use wgpu::{Buffer, CommandEncoder, ComputePipeline, Device, PipelineLayout};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray, PipelineCompound,
    arr_o_gpu::{
        compute_shaders::{
            MATMUL_2D_SHADERS_PATH, MATMUL_ND_CONTIGUOUS_SHADERS_PATH, MATMUL_ND_VIEW_SHADERS_PATH,
        },
        module::method::function::operation::matmul::matmul_operate::MatmulOperate,
    },
    vector_padding,
};

const PIPELINE_MATMUL2D: &'static str = "pipeline_matmul2d";
const PIPELINE_MATMULND_CONTIGUOUS: &'static str = "pipeline_matmulnd_contiguous";
const PIPELINE_MATMULND_VIEW: &'static str = "pipeline_matmulnd_view";

impl ArrOgpuModule {
    pub fn matmul<A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
        B: ArrayCompute,
    {
        let matmul_operate = MatmulOperate::create(array_a, array_b)?;

        let (shape, len, stride, dim, offset, allocated) =
            matmul_operate.create_metadata_output(&mut self.allocator_write());

        let shape_padding: [u32; 8] = vector_padding(shape.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Matmul(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Matmul(err))?
            .try_into()
            .unwrap();

        let pointer = allocated.get_range();
        let metadata_output = self.create_metadata_compound(
            [pointer.0 as u32, pointer.1 as u32],
            len,
            dim,
            offset,
            shape_padding,
            stride_padding,
            stride_padding,
        );

        let wgpu = self.wgpu_init().read().unwrap();
        let read = self.pipeline_cache.read().unwrap();
        let pipeline = if let MatmulOperate::_2D(_, _) = &matmul_operate {
            if let Some(pipeline) = read.get(PIPELINE_MATMUL2D) {
                PipelineCompound::PipelineCache(pipeline)
            } else {
                PipelineCompound::UnsavePipeline(
                    create_pipeline(
                        &wgpu.device,
                        &self.common_pipeline_layout,
                        *self.maximum as f64,
                        MATMUL_2D_SHADERS_PATH,
                    ),
                    PIPELINE_MATMUL2D,
                )
            }
        } else {
            match (
                array_a.check_contiguous_or_view(),
                array_b.check_contiguous_or_view(),
            ) {
                (ArrayType::Contiguous(_), ArrayType::Contiguous(_)) => {
                    if let Some(pipeline) = read.get(PIPELINE_MATMULND_CONTIGUOUS) {
                        PipelineCompound::PipelineCache(pipeline)
                    } else {
                        PipelineCompound::UnsavePipeline(
                            create_pipeline(
                                &wgpu.device,
                                &self.common_pipeline_layout,
                                *self.maximum as f64,
                                MATMUL_ND_CONTIGUOUS_SHADERS_PATH,
                            ),
                            PIPELINE_MATMULND_CONTIGUOUS,
                        )
                    }
                }

                (_, _) => {
                    if let Some(pipeline) = read.get(PIPELINE_MATMULND_VIEW) {
                        PipelineCompound::PipelineCache(pipeline)
                    } else {
                        PipelineCompound::UnsavePipeline(
                            create_pipeline(
                                &wgpu.device,
                                &self.common_pipeline_layout,
                                *self.maximum as f64,
                                MATMUL_ND_VIEW_SHADERS_PATH,
                            ),
                            PIPELINE_MATMULND_VIEW,
                        )
                    }
                }
            }
        };

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Command Encoder For Matmul"),
                });

        set_cache(
            &mut encoder,
            array_a,
            array_b,
            &metadata_output.buffer,
            &self.execute_args,
        )?;

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Matmul"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);
            begin_compute_pass.set_bind_group(0, Some(&self.heap_binding().binding_groups), &[]);

            let (x, y, z) = matmul_operate.get_x_y_z();
            begin_compute_pass.dispatch_workgroups(x, y, z);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let unsave_pipeline = pipeline.get_unsave_pipeline();
        drop(read);
        self.saving_from_pipeline_compound(unsave_pipeline);

        let array = GpuArray {
            length: len as usize,
            metadata_compound: Some(metadata_output),
            module: Arc::new(self.clone()),
            shape,
            stride,
            allocated,
        };

        Ok(array)
    }
}

fn set_cache<A, B>(
    encoder: &mut CommandEncoder,
    array_a: &A,
    array_b: &B,
    metadata_output: &Buffer,
    execute_args: &Arc<Buffer>,
) -> Result<(), ArrOgpuErr>
where
    A: ArrayCompute,
    B: ArrayCompute,
{
    let metadata_a = array_a.metadata_compound().ok_or(ArrOgpuErr::Matmul(
        "Matmul Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    let metadata_b = array_b.metadata_compound().ok_or(ArrOgpuErr::Matmul(
        "Matmul Error, Metadata Not Yet Defined For Array B".to_string(),
    ))?;

    encoder.copy_buffer_to_buffer(&metadata_a.buffer, 0, execute_args, 0, 256);
    encoder.copy_buffer_to_buffer(&metadata_b.buffer, 0, execute_args, 256, 256);
    encoder.copy_buffer_to_buffer(metadata_output, 0, execute_args, 512, 256);

    Ok(())
}

fn create_pipeline(
    device: &Device,
    pipeline_layout: &PipelineLayout,
    heap_len: f64,
    path: &'static str,
) -> ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Matmul"),
        layout: Some(pipeline_layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Matmul"),
            source: wgpu::ShaderSource::Wgsl(path.into()),
        }),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("LEN_HEAP", heap_len - 1.)],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}
