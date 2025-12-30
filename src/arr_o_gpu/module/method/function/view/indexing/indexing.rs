use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArrayView, get_stride_from_shape,
    negative_indexing_converter, vector_padding,
};

impl ArrOgpuModule {
    pub fn index<'a, A>(
        &self,
        array: &'a A,
        index: &[i32],
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        // negative_converter
        let index = negative_indexing_converter(index, &array.shape()).map_err(ArrOgpuErr::from)?;

        // check dim
        let dim = array.dim();
        if index.len() > dim {
            let err = format!(
                "Indexing Out Of Dimension Error, Indexing {:?} but dim of array is {}",
                index, dim
            );
            return Err(ArrOgpuErr::Indexing(err));
        } else if index.is_empty() {
            let err = format!("Idexing Error, Indexing Is Empty");
            return Err(ArrOgpuErr::Indexing(err));
        }
        // check overflow
        let stride = array.stride();
        let shape = array.shape();
        let mut offset_list = vec![0; shape.len()];
        let mut start = array.offset();
        for i in 0..index.len() {
            if index[i] >= shape[i] {
                let err = format!(
                    "Indexing Out Of Range Error, Indexing {:?} but shape of array is {:?}",
                    index, shape
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

        let mut stride = array.stride()[index.len()..].to_vec();
        if stride.is_empty() {
            stride.push(1);
        }

        let binding = self.create_metadata_binding(
            &array.pointer_to_arr(),
            &new_shape,
            &get_stride_from_shape(&new_shape),
            &stride,
            &start,
        );

        let shape_padding: [u32; 10] = vector_padding(new_shape.clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::Indexing(err))?
            .try_into()
            .unwrap();

        let origin_stride_padding: [u32; 10] =
            vector_padding(get_stride_from_shape(&new_shape), 0, 10)
                .map_err(|err| ArrOgpuErr::Indexing(err))?
                .try_into()
                .unwrap();

        let stride_padding: [u32; 10] = vector_padding(stride.clone(), 0, 10)
            .map_err(|err| ArrOgpuErr::Indexing(err))?
            .try_into()
            .unwrap();
        let metadata = self.create_metadata_compound(
            array.pointer_to_arr(),
            array.len(),
            array.dim() as u32,
            start,
            shape_padding,
            stride_padding,
            origin_stride_padding,
        );

        let arr_view = GpuArrayView {
            array,
            pointer: array.pointer(),
            shape: new_shape,
            stride,
            offset: start,
            binding: Some(binding),

            metadata_compound: Some(metadata),
        };

        Ok(arr_view)
    }
}
