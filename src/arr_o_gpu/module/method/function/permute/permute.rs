use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArrayView };

impl ArrOgpuModule {
    pub fn permute<'a, A>(
        &self,
        array: &'a A,
        permute: &[u32]
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
        where A: ArrayView
    {
        let array_shape = array.shape();
        let array_stride = array.stride();

        if array_shape.len() != permute.len() {
            let arr = format!(
                "Permute Error, Array With Form {:?} Cannot Be Permuted By {:?}",
                array_shape,
                permute
            );
            return Err(ArrOgpuErr::Permute(arr));
        }

        let mut over = false;
        let mut repeat = (false, None);
        let out_shape = permute
            .iter()
            .map(|&dim| {
                if (dim as usize) < array_shape.len() {
                    if let None = repeat.1 {
                        repeat = (false, Some(dim));
                        (array_shape[dim as usize], array_stride[dim as usize])
                    } else {
                        repeat = (true, None);
                        (0, 0)
                    }
                } else {
                    over = true;
                    (0, 0)
                }
            })
            .collect::<(Vec<u32>, Vec<u32>)>();

        if over {
            let arr = format!(
                "Permute Error, {:?} Crosses The Array Boundary Of Array {:?}",
                permute,
                array_shape
            );
            return Err(ArrOgpuErr::Permute(arr));
        } else if let Some(_) = repeat.1 {
            let arr = format!("Permute Error, Repeat Index Detected On {:?}", permute);
            return Err(ArrOgpuErr::Permute(arr));
        }

        let array_view = GpuArrayView {
            array: array,
            offset: array.offset(),
            pointer: array.pointer().clone(),
            shape: out_shape.0,
            stride: out_shape.1,
        };

        Ok(array_view)
    }
}
