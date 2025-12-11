use wgpu::{ BufferUsages, wgt::{ BufferDescriptor, CommandEncoderDescriptor } };

use crate::{ ArrayView, GpuArrayView };

impl<'a, A> GpuArrayView<'a, A> where A: ArrayView {
    // unsafe
    pub fn get_heap(&self) -> Vec<f32> {
        let module = self.array.module();
        let wgpu = module.wgpu_init.read().unwrap();
        let heap_buffer = module.heap_buffer();

        // array
        let pointer = self.pointer;
        let len = pointer.1 - pointer.0;
        let mem = std::mem::size_of::<f32>() as u64;
        let size = (len as u64) * mem;

        // copy buffer
        let copy_buffer = wgpu.device.create_buffer(
            &(BufferDescriptor {
                label: Some("create buffer copy for get_heap_pointer"),
                size,
                mapped_at_creation: false,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            })
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For get_heap"),
            })
        );

        let start = (pointer.0 as u64) * mem;
        encoder.copy_buffer_to_buffer(heap_buffer, start, &copy_buffer, 0, size);
        wgpu.queue.submit(Some(encoder.finish()));

        let copy_slice = copy_buffer.slice(..);
        copy_slice.map_async(wgpu::MapMode::Read, |e| e.unwrap());
        wgpu.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let copy = copy_slice.get_mapped_range();
        let data: Vec<f32> = bytemuck::cast_slice(&copy).into();

        drop(copy);
        copy_buffer.unmap();

        data
    }
}
