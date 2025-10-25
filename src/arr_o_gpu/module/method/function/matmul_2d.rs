use std::sync::Arc;

use wgpu::{
    util::{ BufferInitDescriptor, DeviceExt },
    wgt::{ BufferDescriptor, CommandEncoderDescriptor, PollType },
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingResource,
    BindingType,
    BufferBindingType,
    BufferUsages,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    Device,
    MapMode,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    ShaderStages,
};

use crate::{ get_stride_from_shape, ArrOgpuErr, ArrOgpuModule, GpuArray };

impl ArrOgpuModule {
    pub fn matmul_2d(&self, arr_a: GpuArray, arr_b: GpuArray) -> Result<GpuArray, ArrOgpuErr> {
        if arr_a.dim() != 2 || arr_b.dim() != 2 {
            let err = format!(
                "Array matmul 2d Error, dim of array A is {} and dim of array B is {}",
                arr_a.dim(),
                arr_b.dim()
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }

        let m_k = [arr_a.shape[0], arr_a.shape[1]];
        let k_n = [arr_b.shape[0], arr_b.shape[1]];
        if m_k[1] != k_n[0] {
            let err = format!(
                "Array matmul 2d Error, Array A {:?} can't matmul with Array B {:?}",
                arr_a.shape(),
                arr_b.shape()
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }
        let out_shape = [m_k[0], k_n[1]];

        let wgpu_init = &self.wgpu_init.read().unwrap();
        let binding_layout = build_binding_layout(&wgpu_init.device);
        let binding_layout_of_output = build_binding_of_output_layout(&wgpu_init.device);

        // Array A
        // pointer
        let pointer_a = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&arr_a.pointer_to_arr()),
            })
        );

        // shape
        let shape_a = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&arr_a.shape),
            })
        );

        // stride
        let stride_a = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&arr_a.stride),
            })
        );

        // Array B
        // pointer
        let pointer_b = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&arr_b.pointer_to_arr()),
            })
        );

        // shape
        let shape_b = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&arr_b.shape),
            })
        );

        // stride
        let stride_b = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&arr_b.stride),
            })
        );

        // output
        // size
        let mem_f32 = std::mem::size_of::<f32>() as u64;
        let len = out_shape.iter().product::<u32>();
        let size = (len as u64) * mem_f32;
        let output = wgpu_init.device.create_buffer(
            &(BufferDescriptor {
                label: Some("Create Buffer Output For Matmul 2D"),
                mapped_at_creation: false,
                size,
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            })
        );

        // pointer
        let (type_output_pointer, output_pointer_start, output_pointer_end) = self
            .allocator_write()
            .pointer_input(len);
        let output_pointer_list = [output_pointer_start, output_pointer_end];
        let output_pointer_buffer = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Buffer Output Pointer For Matmul 2D"),
                contents: bytemuck::cast_slice(&output_pointer_list),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            })
        );

        // stride
        let stride_out = out_shape.to_vec();
        let stride_out = get_stride_from_shape(&stride_out);
        let stride_out = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Stride Output Buffer Layout For Matmul 2D"),
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&stride_out),
            })
        );

        // copy
        // let copy_buffer = wgpu_init.device.create_buffer(
        //     &(BufferDescriptor {
        //         label: Some("Create Copy Buffer For Matmul 2D"),
        //         mapped_at_creation: false,
        //         size,
        //         usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        //     })
        // );

        let binding = wgpu_init.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create Binding For Matmul 2D"),
                layout: &binding_layout,
                entries: &[
                    // array A
                    // pointer
                    BindGroupEntry {
                        binding: 0,
                        resource: pointer_a.as_entire_binding(),
                    },
                    // shape
                    BindGroupEntry {
                        binding: 1,
                        resource: shape_a.as_entire_binding(),
                    },
                    // stride
                    BindGroupEntry {
                        binding: 2,
                        resource: stride_a.as_entire_binding(),
                    },
                    // Array B
                    // pointer
                    BindGroupEntry {
                        binding: 3,
                        resource: pointer_b.as_entire_binding(),
                    },
                    // shape
                    BindGroupEntry {
                        binding: 4,
                        resource: shape_b.as_entire_binding(),
                    },
                    // stride
                    BindGroupEntry {
                        binding: 5,
                        resource: stride_b.as_entire_binding(),
                    },
                ],
            })
        );

        let binding_of_output = wgpu_init.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create Binding OF Output For Matmul 2D"),
                layout: &binding_layout_of_output,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output_pointer_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: stride_out.as_entire_binding(),
                    },
                ],
            })
        );

        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Matmul 2D"),
            source: ShaderSource::Wgsl(
                include_str!("./../../../shader/shaders/matmul_2d.wgsl").into()
            ),
        });

        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Matmul 2D"),
                push_constant_ranges: &[],
                bind_group_layouts: &[
                    &self.binding_compounds.read().unwrap()[0].binding_group_layouts,
                    &binding_layout,
                    &binding_layout_of_output,
                ],
            })
        );

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Matmul 2D"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Matmul 2D"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Matmul 2D"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);

            // group 0 binding 0
            let heap_binding = &self.binding_compounds().read().unwrap()[0].binding_groups;
            bcp.set_bind_group(0, Some(heap_binding), &[]);

            // group 1 binding 0 - 5
            bcp.set_bind_group(1, Some(&binding), &[]);

            // group 2
            bcp.set_bind_group(2, Some(&binding_of_output), &[]);

            let m = ((out_shape[0] as f32) / 16.0).ceil() as u32;
            let n = ((out_shape[1] as f32) / 16.0).ceil() as u32;
            // m, n, k
            bcp.dispatch_workgroups(m, n, 1);
        }

        // encoder.copy_buffer_to_buffer(&output, 0, &copy_buffer, 0, size);

        wgpu_init.queue.submit(Some(encoder.finish()));

        // let slice_buffer = copy_buffer.slice(..);
        // slice_buffer.map_async(MapMode::Read, |res| res.unwrap());
        wgpu_init.device.poll(PollType::Wait).unwrap();

        // let data_buffer = slice_buffer.get_mapped_range();
        // let data: Vec<f32> = bytemuck::cast_slice(&data_buffer).into();

        // let array = self.array_from_vector(&data, &out_shape).unwrap();

        let pointer = (output_pointer_start as usize, output_pointer_end as usize);

        let len = out_shape.iter().product::<u32>() as usize;
        let stride = get_stride_from_shape(&out_shape);
        let array = GpuArray {
            module: Arc::new(self.clone()),
            shape: out_shape.to_vec(),
            length: len,
            pointer,
            space_type: type_output_pointer,
            stride,
        };

        Ok(array)
    }
}

fn build_binding_layout(device: &Device) -> wgpu::BindGroupLayout {
    let binding_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Binding Layout For Matmul 2D"),
            entries: &[
                // array A
                // pointer
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // shape
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // stride
                BindGroupLayoutEntry {
                    binding: 2,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // array B
                // pointer
                BindGroupLayoutEntry {
                    binding: 3,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // shape
                BindGroupLayoutEntry {
                    binding: 4,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // stride
                BindGroupLayoutEntry {
                    binding: 5,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        })
    );
    binding_layout
}

fn build_binding_of_output_layout(device: &Device) -> BindGroupLayout {
    let binding_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Binding Layout Of Output For Matmul 2D"),
            entries: &[
                // output
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // pointer_output
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // stride_output
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        })
    );
    binding_layout
}
