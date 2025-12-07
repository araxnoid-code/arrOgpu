use wgpu::{ wgt::{ BufferDescriptor, CommandEncoderDescriptor, PollType }, BufferUsages };

use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArray, GpuArrayView };

impl ArrOgpuModule {
    pub fn index(&self, arr: &GpuArray, index: &[u32]) -> Result<GpuArray, ArrOgpuErr> {
        let dim = arr.dim();
        // check dim
        if index.len() > dim || index.is_empty() {
            let err = format!(
                "Indexing Out Of Dimension Error, Indexing {:?} but dim of array is {}",
                index,
                dim
            );
            return Err(ArrOgpuErr::Indexing(err));
        }
        // check overflow
        let stride = &arr.stride;
        let shape = &arr.shape;
        let mut start = 0;
        for i in 0..index.len() {
            if index[i] >= shape[i] {
                let err = format!(
                    "Indexing Out Of Range Error, Indexing {:?} but shape of array is {:?}",
                    index,
                    shape
                );
                return Err(ArrOgpuErr::Indexing(err));
            } else {
                let idx = index[i];
                start += idx * stride[i];
            }
        }

        let end = start + stride[index.len() - 1];

        // get range on GPU
        let wgpu_init = &self.wgpu_init.read().unwrap();

        let f32_size = std::mem::size_of::<f32>() as u64;
        let size = ((end - start) as u64) * f32_size;
        let copy_buffer = wgpu_init.device.create_buffer(
            &(BufferDescriptor {
                label: Some("create buffer copy for indexing"),
                mapped_at_creation: false,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                size,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder for Indexing"),
            })
        );

        // copy heap
        let heap = self.heap_buffer();
        encoder.copy_buffer_to_buffer(heap, (start as u64) * f32_size, &copy_buffer, 0, size);

        wgpu_init.queue.submit(Some(encoder.finish()));

        let slice_buffer = copy_buffer.slice(..);
        slice_buffer.map_async(wgpu::MapMode::Read, |v| v.unwrap());
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let data_buffer = slice_buffer.get_mapped_range();
        let data: Vec<f32> = bytemuck::cast_slice(&data_buffer).to_vec();

        drop(data_buffer);
        copy_buffer.unmap();

        let new_shape = if index.len() == shape.len() {
            vec![1]
        } else {
            shape[index.len()..].to_vec()
        };

        let array = self.array_from_vector(&data, &new_shape).unwrap();

        Ok(array)
    }

    pub fn index_view<'a, A>(
        &self,
        arr: &'a A,
        index: &[u32]
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
        where A: ArrayView
    {
        let dim = arr.dim();
        // check dim
        if index.len() > dim || index.is_empty() {
            let err = format!(
                "Indexing Out Of Dimension Error, Indexing {:?} but dim of array is {}",
                index,
                dim
            );
            return Err(ArrOgpuErr::Indexing(err));
        }
        // check overflow
        let stride = arr.stride();
        let shape = arr.shape();
        let mut offset_list = vec![0; shape.len()];
        let mut start = arr.offset();
        for i in 0..index.len() {
            if index[i] >= shape[i] {
                let err = format!(
                    "Indexing Out Of Range Error, Indexing {:?} but shape of array is {:?}",
                    index,
                    shape
                );
                return Err(ArrOgpuErr::Indexing(err));
            } else {
                let idx = index[i];
                start += idx * stride[i];
                offset_list[i] = idx * stride[i];
            }
        }

        let new_shape = if index.len() == shape.len() {
            vec![1]
        } else {
            shape[index.len()..].to_vec()
        };

        let mut stride = arr.stride()[index.len()..].to_vec();
        if stride.is_empty() {
            stride.push(1);
        }

        let arr_view = GpuArrayView {
            array: arr,
            pointer: arr.pointer(),
            shape: new_shape,
            stride: stride,
            offset: start,
        };

        Ok(arr_view)
    }
}
