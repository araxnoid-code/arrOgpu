use std::collections::HashSet;

use wgpu::ShaderModuleDescriptor;

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
