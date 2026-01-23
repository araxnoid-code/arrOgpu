use wgpu::{
    BufferUsages, MapMode,
    wgt::{BufferDescriptor, CommandEncoderDescriptor, PollType},
};

use crate::ArrOgpuModule;

impl ArrOgpuModule {
    pub fn get_heap(&self) -> Vec<f32> {
        let wgpu_init = &self.wgpu_init.read().unwrap();
        let size = (std::mem::size_of::<f32>() * (*self.maximum as usize)) as u64;

        let copy_buffer = wgpu_init.device.create_buffer(
            &(BufferDescriptor {
                label: Some("create copy buffer for get_heap"),
                mapped_at_creation: false,
                size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            }),
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("create encoder for get_heap"),
            }),
        );

        encoder.copy_buffer_to_buffer(&self.heap_buffer, 0, &copy_buffer, 0, size);
        let index = wgpu_init.queue.submit(Some(encoder.finish()));

        let buffer_slice = copy_buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |e| e.unwrap());
        wgpu_init.device.poll(PollType::Wait { submission_index: Some(index), timeout: None }).unwrap();

        let data = buffer_slice.get_mapped_range();
        let heap: Vec<f32> = bytemuck::cast_slice(&data).into();
        drop(data);
        copy_buffer.unmap();

        heap
    }
}
