use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BufferBindingType,
    BufferUsages,
    Device,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

use crate::{
    ArrOgpuModule,
    ArrayView,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::PointerOption,
};

impl ArrOgpuModule {
    pub(crate) fn add_skalar<A>(&self, array: &A, pointer_option: PointerOption) where A: ArrayView {
        let wgpu = self.wgpu_init.read().unwrap();

        // output metadata
        let shape = array.shape();
        let len = array.len();
        let stride = array.stride();
        let allocate = self.allocator_write().pointer_input(len);

        // binding
        // // heap
        let heap_bind = &self.heap_binding;
        // // array
        let array_bind = array.binding();
        // // output
        let out_bind = self.array_data_binding(
            &[allocate.1, allocate.2],
            &shape,
            &stride,
            &stride,
            &0
        );
        // // scalar
        let scalar_bind = scalar_binding(&wgpu.device, pointer_option);
    }
}

fn scalar_binding(
    device: &Device,
    pointer_option: PointerOption
) -> (BindGroupLayout, wgpu::BindGroup) {
    // buffer
    let scalar_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Buffer Scalar For Add Scalar"),
            usage: BufferUsages::UNIFORM,
            contents: match &pointer_option {
                PointerOption::Pointer(pointer) => bytemuck::cast_slice(pointer),
                PointerOption::Skalar(scalar) => bytemuck::bytes_of(scalar),
            },
        })
    );

    // bind_group_layout
    let bind_group_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Bind Group Layout For Add Scalar"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
            ],
        })
    );

    // bind_group
    let bind_group = device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("Create Bind Group Layout For Add Scalar"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: scalar_buffer.as_entire_binding(),
                },
            ],
        })
    );

    (bind_group_layout, bind_group)
}
