use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArrayView, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn broadcast_view<'a, A>(
        &self,
        array: &'a A,
        broadcast: &[u32]
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
        where A: ArrayView
    {
        let arr_shape = array.shape();
        // expend shape
        let diff_range = broadcast.len() - arr_shape.len();
        let extend_arr_shape = if diff_range != 0 {
            let mut extend = vec![1;diff_range];
            extend.extend_from_slice(arr_shape);
            extend
        } else {
            arr_shape.clone()
        };

        let out_shape = broadcast.to_vec();
        let mut out_stride = get_stride_from_shape(&extend_arr_shape);
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

        let array_view = GpuArrayView {
            array,
            offset: array.offset(),
            pointer: array.pointer(),
            shape: out_shape,
            stride: out_stride,
        };

        Ok(array_view)
    }
}
