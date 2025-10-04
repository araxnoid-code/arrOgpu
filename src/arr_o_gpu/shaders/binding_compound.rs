use wgpu::{ BindGroup, BindGroupLayout };

pub struct BindGroupCompound {
    pub group: u32,
    pub binding_group_layouts: BindGroupLayout,
    pub binding_groups: BindGroup,
}
