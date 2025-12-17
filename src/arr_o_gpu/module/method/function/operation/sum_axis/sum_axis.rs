use std::collections::HashSet;

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    Device,
    ShaderModuleDescriptor,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn sum_axis<A, B>(&self, array_a: &A, axis: &[usize]) -> Result<(), ArrOgpuErr>
        where A: ArrayView
    {
        let mut axis = axis.to_vec();
        axis.sort();

        // error handling
        error_handling(array_a, &axis)?;

        // out meta data
        let mut out_shape = array_a.shape().clone();
        axis.iter()
            .rev()
            .for_each(|idx| {
                out_shape.remove(*idx);
            });
        let stride = get_stride_from_shape(&out_shape);
        let out_len = out_shape.iter().product::<u32>();
        let allocate = self.allocator.write().unwrap().pointer_input(out_len);

        // binding
        let heap_binding = &self.heap_binding;
        let array_binding = array_a.binding();
        let out_binding = self.array_data_binding(
            &[allocate.1, allocate.2],
            &out_shape,
            &stride,
            &stride,
            &0
        );

        // others

        // wgpu
        let wgpu = self.wgpu_init.read().unwrap();

        // pipeline
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Created Shader Module For Sum Axis"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sum_axis.wgsl").into()),
        });

        Ok(())
    }
}

fn error_handling<A>(array_a: &A, axis: &[usize]) -> Result<(), ArrOgpuErr> where A: ArrayView {
    // indexing out of shape
    let shape = array_a.shape();
    let axis_len = axis.len();
    if axis_len > shape.len() {
        let error = format!(
            "Sum Axis Error, Indexing On {:?} Exceeds The Dimension Of Array {:?}",
            axis,
            array_a.shape()
        );
        return Err(ArrOgpuErr::SumAxis(error));
    }

    // repeated index
    let mut value: HashSet<usize> = HashSet::new();
    for idx in axis {
        if let None = value.get(idx) {
            value.insert(*idx);
        } else {
            let error = format!(" Sum Axis Error, Found Repeated Indexes On The Axis {:?}", axis);
            return Err(ArrOgpuErr::SumAxis(error));
        }
    }

    Ok(())
}

fn others_binding(
    device: &Device,
    axis_list: &[u32],
    shape_of_slice: &[u32],
    stride_of_slice: &[u32]
) -> (BindGroupLayout, wgpu::BindGroup) {
    // buffer
    // // axis_list
    let axis_list_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Axis List Buffer For Init For Sum Axis"),
            contents: bytemuck::cast_slice(axis_list),
            usage: BufferUsages::STORAGE,
        })
    );

    // // shape_of_slice
    let shape_of_slice_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Axis List Buffer For Init For Sum Axis"),
            contents: bytemuck::cast_slice(shape_of_slice),
            usage: BufferUsages::STORAGE,
        })
    );

    // // stride_of_slice
    let stride_of_slice_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Axis List Buffer For Init For Sum Axis"),
            contents: bytemuck::cast_slice(stride_of_slice),
            usage: BufferUsages::STORAGE,
        })
    );

    // binding layout
    let binding_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Others Binding Layout For Sum Axis"),
            entries: &[
                // axis_list
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },

                // shape_of_slice
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },

                // stride_of_slice
                BindGroupLayoutEntry {
                    binding: 2,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
            ],
        })
    );

    let binding = device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("Create Others Binding For Sum Axis"),
            layout: &binding_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: axis_list_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: shape_of_slice_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: stride_of_slice_buffer.as_entire_binding(),
                },
            ],
        })
    );

    (binding_layout, binding)
}
