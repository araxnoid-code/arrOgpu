use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArrayView,
    arr_o_gpu::module::method::function::view::slicing::slice_range::SliceRange,
    get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub fn slicing<'a, A>(
        &self,
        array: &'a A,
        slice: &[SliceRange],
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        if array.shape().len() < slice.len() || slice.len() == 0 {
            let err = format!(
                "Array Slicing Error, Array {:?} can't Slice By {:?} cause out of range",
                array.shape(),
                slice
            );

            return Err(ArrOgpuErr::Slicing(err));
        }

        let stride = array.stride();
        let mut output_shape = vec![];
        let mut offset = array.offset();
        for (i, _) in array.shape().iter().enumerate() {
            if let Some(range) = slice.get(i) {
                let start = range.start.unwrap_or(0);
                let end = range.end.unwrap_or(array.shape()[i]);

                if start >= end || end > array.shape()[i] {
                    let err = format!(
                        "Array Slicing Error, Error detected for {:?} in slice {:?}",
                        range, slice
                    );
                    return Err(ArrOgpuErr::Slicing(err));
                }

                output_shape.push((end - start) as u32);
                offset += start * stride[i];
            } else {
                output_shape.push(array.shape()[i] as u32);
            }
        }

        let binding = self.create_metadata_binding(
            &array.pointer_to_arr(),
            &output_shape,
            &get_stride_from_shape(&output_shape),
            array.stride(),
            &offset,
        );

        let shape_padding: [u32; 10] = vector_padding(output_shape.clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::Slicing(err))?
            .try_into()
            .unwrap();

        let origin_stride_padding: [u32; 10] =
            vector_padding(get_stride_from_shape(&output_shape), 0, 10)
                .map_err(|err| ArrOgpuErr::Slicing(err))?
                .try_into()
                .unwrap();

        let stride_padding: [u32; 10] = vector_padding(array.stride().clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::Slicing(err))?
            .try_into()
            .unwrap();

        let metadata = self.create_metadata_compound(
            array.pointer_to_arr(),
            array.len(),
            array.dim() as u32,
            offset,
            shape_padding,
            stride_padding,
            origin_stride_padding,
        );

        let array_view = GpuArrayView {
            array: array,
            offset,
            pointer: array.pointer(),
            shape: output_shape,
            stride: array.stride().clone(),
            binding: Some(binding),
            metadata_compound: Some(metadata),
        };

        Ok(array_view)
    }
}
