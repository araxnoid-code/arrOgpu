use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArrayView };

impl ArrOgpuModule {
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
