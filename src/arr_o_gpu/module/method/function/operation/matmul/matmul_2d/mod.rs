use std::sync::{Arc, RwLockWriteGuard};

use wgpu::{Buffer, CommandEncoder, ComputePipeline, Device, PipelineLayout};

use crate::{
    Allocator, ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray, PipelineCompound,
    arr_o_gpu::compute_shaders::{MATMUL_2D_SHADERS_PATH, MATMUL_ND_SHADERS_PATH},
    get_stride_from_shape, vector_padding,
};

mod matmul_2d;

const PIPELINE_MATMUL2D: &'static str = "pipeline_matmul2d";
const PIPELINE_MATMULND: &'static str = "pipeline_matmulnd";

enum MatmulOperate<'a, A, B>
where
    A: ArrayCompute,
    B: ArrayCompute,
{
    _2D(&'a A, &'a B),
    ND(&'a A, &'a B),
}

impl<'a, A, B> MatmulOperate<'a, A, B>
where
    A: ArrayCompute,
    B: ArrayCompute,
{
    pub fn create(array_a: &'a A, array_b: &'a B) -> Result<MatmulOperate<'a, A, B>, ArrOgpuErr> {
        let dim = array_a.dim();
        if array_a.shape().len() <= 1 || array_b.shape().len() <= 1 {
            let err = format!(
                "Array Matmul 2d Error, Shape Of Array A Is {:?} And Shape Of Array B Is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            Err(ArrOgpuErr::Matmul2D(err))
        } else if array_a.shape().len() == 2
            && array_b.shape().len() == 2
            && array_a.shape()[1] == array_b.shape()[0]
        {
            Ok(Self::_2D(array_a, array_b))
        } else if array_a.shape().len() == array_b.shape().len()
            && array_a.shape()[..dim - 2] == array_b.shape()[..dim - 2]
            && array_a.shape()[dim - 1] == array_b.shape()[dim - 2]
        {
            Ok(Self::ND(array_a, array_b))
        } else {
            let err = format!(
                "Array Matmul 2d Error, Shape Of Array A Is {:?} And Shape Of Array B Is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            Err(ArrOgpuErr::Matmul2D(err))
        }
    }

    fn create_metadata_output(
        &self,
        allocate: &mut RwLockWriteGuard<'_, Allocator>,
    ) -> (
        Vec<u32>,
        u32,
        Vec<u32>,
        u32,
        u32,
        (crate::SpaceType, u32, u32),
    ) {
        match self {
            Self::_2D(arr_a, arr_b) => {
                let shape = vec![arr_a.shape()[0], arr_b.shape()[1]];
                let len = shape.iter().product::<u32>();
                let stride = get_stride_from_shape(&shape);
                let dim = 2;
                let offset = 0;
                let allocate = allocate.pointer_input(len);

                (shape, len, stride, dim, offset, allocate)
            }

            Self::ND(arr_a, arr_b) => {
                let mut shape = arr_a.shape().clone();
                if let Some(coll) = shape.last_mut() {
                    *coll = *arr_b.shape().last().unwrap();
                }
                let len = shape.iter().product::<u32>();
                let stride = get_stride_from_shape(&shape);
                let dim = arr_a.dim() as u32;
                let offset = 0;
                let allocate = allocate.pointer_input(len);

                (shape, len, stride, dim, offset, allocate)
            }
        }
    }
}

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

        let matmul_operate = MatmulOperate::create(array_a, array_b)?;

        let (shape, len, stride, dim, offset, allocate) =
            matmul_operate.create_metadata_output(&mut self.allocator_write());

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
        let pipeline = if let MatmulOperate::_2D(_, _) = &matmul_operate {
            if let Some(pipeline) = read.get(PIPELINE_MATMUL2D) {
                PipelineCompound::PipelineCache(pipeline)
            } else {
                PipelineCompound::Pipeline(
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
            if let Some(pipeline) = read.get(PIPELINE_MATMULND) {
                PipelineCompound::PipelineCache(pipeline)
            } else {
                PipelineCompound::Pipeline(
                    create_pipeline(
                        &wgpu.device,
                        &self.common_pipeline_layout,
                        *self.maximum as f64,
                        MATMUL_ND_SHADERS_PATH,
                    ),
                    PIPELINE_MATMULND,
                )
            }
        };

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Command Encoder For Matmul2d"),
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

        // wgpu.device
        //     .poll(wgpu::wgt::PollType::Wait {
        //         submission_index: Some(id),
        //         timeout: None,
        //     })
        //     .unwrap();

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
    path: &'static str,
) -> ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Matmul2d"),
        layout: Some(pipeline_layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Matmul2d"),
            source: wgpu::ShaderSource::Wgsl(path.into()),
        }),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("LEN_HEAP", heap_len)],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}
