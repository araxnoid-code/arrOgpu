use std::ops::{ Range, RangeBounds };

use wgpu::{ wgt::{ BufferDescriptor, CommandEncoderDescriptor, PollType }, BufferUsages, MapMode };

use crate::GpuArray;

impl GpuArray {
    pub fn get_heap(&self) -> Vec<f32> {
        let module = &self.module;
        let wgpu_init = &module.wgpu_init.read().unwrap();
        let heap_buffer = &module.heap_buffer;

        let pointer = self.pointer;
        let len = pointer.1 - pointer.0;
        let mem = std::mem::size_of::<f32>();
        let size = (mem * len) as u64;

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

        let start = pointer.0 * mem;
        encoder.copy_buffer_to_buffer(heap_buffer, start as u64, &copy_buffer, 0, size);
        wgpu_init.queue.submit(Some(encoder.finish()));

        let buffer_slice = copy_buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |e| e.unwrap());
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).into();
        drop(data);
        copy_buffer.unmap();

        result
    }
}
