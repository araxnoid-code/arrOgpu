use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder, ComputePipeline, Device, PipelineLayout};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray, PipelineCompound,
    arr_o_gpu::compute_shaders::MATMUL_CONTIGUOUS_SHADERS_PATH, get_stride_from_shape,
    vector_padding,
};

mod matmul_2d;

const PIPELINE_MATMUL2D: &'static str = "pipeline_matmul2d_contiguous";

impl ArrOgpuModule {
    pub fn matmul_2d_metadata<A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
        B: ArrayCompute,
    {
        // error
        if (array_a.dim() != 2 || array_a.dim() != 2) || (array_a.shape()[1] != array_b.shape()[0])
        {
            let err = format!(
                "Array Matmul 2d Error, Shape Of Array A Is {:?} And Shape Of Array B Is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }
        // handling

        // metadata_output
        let shape = vec![array_a.shape()[0], array_b.shape()[1]];
        let len = shape.iter().product::<u32>();
        let stride = get_stride_from_shape(&shape);
        let dim = 2;
        let offset = 0;
        let allocate = self.allocator_write().pointer_input(len);

        let shape_padding: [u32; 8] = vector_padding(shape.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Matmul2D(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Matmul2D(err))?
            .try_into()
            .unwrap();

        let metadata_output = self.create_metadata_compound(
            [allocate.1, allocate.2],
            len,
            dim,
            offset,
            shape_padding,
            stride_padding,
            stride_padding,
        );

        let wgpu = self.wgpu_init().read().unwrap();
        let read = self.pipeline_cache.read().unwrap();

        let pipeline = if let Some(pipeline) = read.get(PIPELINE_MATMUL2D) {
            PipelineCompound::PipelineCache(pipeline)
        } else {
            PipelineCompound::Pipeline(create_pipeline(
                &wgpu.device,
                &self.common_pipeline_layout,
                *self.maximum as f64,
            ))
        };

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create COmmand Encoder For Matmul2d"),
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
                label: Some("Create Begin Compute Pass For Matmul 2d"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);
            begin_compute_pass.set_bind_group(0, Some(&self.heap_binding().binding_groups), &[]);

            let x = (array_a.shape()[0] + 15) / 16;
            let y = (array_b.shape()[1] + 15) / 16;
            begin_compute_pass.dispatch_workgroups(x, y, 1);
        }

        let id = wgpu.queue.submit(Some(encoder.finish()));
        wgpu.device
            .poll(wgpu::wgt::PollType::Wait {
                submission_index: Some(id),
                timeout: None,
            })
            .unwrap();

        let array = GpuArray {
            pointer: (allocate.1, allocate.2),
            binding: None,
            length: len as usize,
            metadata_compound: Some(metadata_output),
            module: Arc::new(self.clone()),
            shape,
            stride,
            space_type: allocate.0,
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
    let metadata_a = array_a.metadata_compound().ok_or(ArrOgpuErr::Matmul2D(
        "Matmul 2d Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    let metadata_b = array_b.metadata_compound().ok_or(ArrOgpuErr::Matmul2D(
        "Matmul 2d Error, Metadata Not Yet Defined For Array B".to_string(),
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
) -> ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Matmul2d"),
        layout: Some(pipeline_layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Matmul2d"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_CONTIGUOUS_SHADERS_PATH.into()),
        }),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("LEN_HEAP", heap_len)],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}
