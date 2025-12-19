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
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

pub(crate) fn others_binding(
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
