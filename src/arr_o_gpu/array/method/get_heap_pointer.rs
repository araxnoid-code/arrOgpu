use std::ops::{ Range, RangeBounds };

use wgpu::{ wgt::{ BufferDescriptor, CommandEncoderDescriptor, PollType }, BufferUsages };

use crate::GpuArray;

impl<'a> GpuArray<'a> {
    pub fn get_heap_pointer(&self) -> Vec<f32> {
        let module = &self.module.write().unwrap();
        let wgpu_init = &module.wgpu_init;
        let heap_buffer = &module.heap_buffer;

        let pointer = self.pointer;
        let len = pointer.1 - pointer.0;
        println!("{}", len);
        let size = (std::mem::size_of::<f32>() * len) as u64;
        let copy_buffer = wgpu_init.device.create_buffer(
            &(BufferDescriptor {
                label: Some("create buffer copy for get_heap_pointer"),
                mapped_at_creation: false,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                size,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("create encoder for get_heap_pointer"),
            })
        );

        encoder.copy_buffer_to_buffer(heap_buffer, 0, &copy_buffer, 0, size);

        let range = pointer.0 as u64..pointer.1 as u64;
        let buffer_slice = copy_buffer.slice(range);
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).into();
        drop(data);
        copy_buffer.unmap();

        result
    }
}
