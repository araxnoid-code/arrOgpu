use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArrayView, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn reshape<'a, A>(
        &self,
        array: &'a A,
        shape: &[u32]
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
        where A: ArrayView
    {
        let length_of_new_shape = shape.iter().product::<u32>();
        let len_arr = array.len();

        if shape.is_empty() {
            let err = "Reshape Error, shape input is empty".to_string();
            return Err(ArrOgpuErr::Reshape(err));
        } else if length_of_new_shape != len_arr {
            let err = format!(
                "Reshape Error, the shape {:?} does not correspond to an array that has length {}",
                shape,
                len_arr
            );
            return Err(ArrOgpuErr::Reshape(err));
        } else if array.offset() != 0 || !array.is_contiguous() {
            let err = format!("Reshape Error, Array No Contiguous");
            return Err(ArrOgpuErr::Reshape(err));
        }

        let shape = shape.to_vec();
        let array_view = GpuArrayView {
            array: array,
            offset: array.offset(),
            pointer: array.pointer(),
            stride: get_stride_from_shape(&shape),
            shape,
        };

        Ok(array_view)
    }
}
