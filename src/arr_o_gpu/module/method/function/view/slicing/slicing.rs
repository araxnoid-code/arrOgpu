use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArrayView,
    arr_o_gpu::module::method::function::view::slicing::slice_range::SliceRange,
    get_stride_from_shape,
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

        let binding = self.array_data_binding(
            &array.pointer_to_arr(),
            &output_shape,
            &get_stride_from_shape(&output_shape),
            array.stride(),
            &offset,
        );

        let array_view = GpuArrayView {
            array: array,
            offset,
            pointer: array.pointer(),
            shape: output_shape,
            stride: array.stride().clone(),
            binding,
        };

        Ok(array_view)
    }
}
