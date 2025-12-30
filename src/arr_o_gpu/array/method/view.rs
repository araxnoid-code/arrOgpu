use crate::{ArrOgpuErr, GpuArray, GpuArrayView, vector_padding};

impl GpuArray {
    pub fn view(&self) -> Result<GpuArrayView<'_, GpuArray>, ArrOgpuErr> {
        let shape = self.shape.clone();

        let binding = self.module.create_metadata_binding(
            &self.pointer_to_arr(),
            &shape,
            &self.stride,
            &self.stride,
            &0,
        );

        let padding_shape: [u32; 10] = (vector_padding(shape.clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::Padding(err))?)
        .try_into()
        .unwrap();

        let padding_stride: [u32; 10] = vector_padding(self.stride().clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::Padding(err))?
            .try_into()
            .unwrap();

        let metadata = self.module().create_metadata_compound(
            self.pointer_to_arr(),
            self.len() as u32,
            self.dim() as u32,
            0,
            padding_shape,
            padding_stride,
            padding_stride,
        );

        Ok(GpuArrayView {
            array: self,
            pointer: self.pointer,
            stride: self.stride.clone(),
            shape,
            offset: 0,
            binding: Some(binding),
            metadata_compound: Some(metadata),
        })
    }
}
