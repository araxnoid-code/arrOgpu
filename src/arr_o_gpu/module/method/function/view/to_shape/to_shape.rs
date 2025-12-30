use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArrayView, get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub fn to_shape<'a, A>(
        &self,
        array: &'a A,
        shape: &[u32],
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let length_of_new_shape = shape.iter().product::<u32>();
        let len_arr = array.len();

        if shape.is_empty() {
            let err = "Reshape Error, shape input is empty".to_string();
            return Err(ArrOgpuErr::ToShape(err));
        } else if length_of_new_shape != len_arr {
            let err = format!(
                "Reshape Error, the shape {:?} does not correspond to an array that has length {}",
                shape, len_arr
            );
            return Err(ArrOgpuErr::ToShape(err));
        } else if array.offset() != 0 || !array.is_contiguous() {
            let err = format!("Reshape Error, Array No Contiguous");
            return Err(ArrOgpuErr::ToShape(err));
        }

        let shape = shape.to_vec();
        let stride = get_stride_from_shape(&shape);
        let binding = self.create_metadata_binding(
            &array.pointer_to_arr(),
            &shape,
            &stride,
            &stride,
            &array.offset(),
        );

        let shape_padding: [u32; 10] = vector_padding(shape.clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::ToShape(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 10] = vector_padding(stride.clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::ToShape(err))?
            .try_into()
            .unwrap();

        let metadata = self.create_metadata_compound(
            array.pointer_to_arr(),
            array.len(),
            array.dim() as u32,
            array.offset(),
            shape_padding,
            stride_padding,
            stride_padding,
        );

        let array_view = GpuArrayView {
            array: array,
            offset: array.offset(),
            pointer: array.pointer(),
            stride,
            shape,
            binding: Some(binding),
            metadata_compound: Some(metadata),
        };

        Ok(array_view)
    }
}
