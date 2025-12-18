use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArrayView, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn broadcast<'a, A>(
        &self,
        array: &'a A,
        broadcast: &[u32]
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
        where A: ArrayView
    {
        let arr_shape = array.shape();
        if broadcast.len() < arr_shape.len() {
            let err = format!(
                "Array Broadcasting Error, Array {:?} can't Broadcast To {:?}",
                arr_shape,
                broadcast
            );

            return Err(ArrOgpuErr::Broadcast(err));
        }

        // expend shape
        let diff_range = broadcast.len() - arr_shape.len();
        let (extend_arr_shape, mut out_stride) = if diff_range != 0 {
            let mut extend = vec![1;diff_range];
            extend.extend_from_slice(arr_shape);

            let mut stride_extend = vec![0;diff_range];
            stride_extend.extend_from_slice(array.stride());
            (extend, stride_extend)
        } else {
            (arr_shape.clone(), array.stride().clone())
        };

        for i in (0..broadcast.len()).rev() {
            let arr_dim = extend_arr_shape[i];
            let broadcast_dim = broadcast[i];

            if arr_dim != broadcast_dim {
                if arr_dim == 1 {
                    out_stride[i as usize] = 0;
                } else {
                    let err = format!(
                        "Array Broadcasting Error, Array {:?} can't Broadcast to {:?}",
                        arr_shape,
                        broadcast
                    );

                    return Err(ArrOgpuErr::Broadcast(err));
                }
            }
        }

        let out_shape = broadcast.to_vec();
        let binding = self.array_data_binding(
            &array.pointer_to_arr(),
            &out_shape,
            &get_stride_from_shape(&out_shape),
            &out_stride,
            &array.offset()
        );

        let array_view = GpuArrayView {
            array,
            offset: array.offset(),
            pointer: array.pointer(),
            shape: out_shape,
            stride: out_stride,
            binding,
        };

        Ok(array_view)
    }
}
