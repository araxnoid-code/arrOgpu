use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArrayView, get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub fn permute<'a, A>(
        &self,
        array: &'a A,
        permute: &[u32],
    ) -> Result<GpuArrayView<'a, A>, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let array_shape = array.shape();
        let array_stride = array.stride();

        if array_shape.len() != permute.len() {
            let arr = format!(
                "Permute Error, Array With Form {:?} Cannot Be Permuted By {:?}",
                array_shape, permute
            );
            return Err(ArrOgpuErr::Permute(arr));
        }

        let mut over = false;
        let mut repeat = (false, None);
        let out_shape = permute
            .iter()
            .map(|&dim| {
                if (dim as usize) < array_shape.len() {
                    if let Some(r) = repeat.1 {
                        if r == dim {
                            repeat = (true, None);
                        }
                        repeat.1 = Some(dim);
                    } else if !repeat.0 {
                        repeat = (false, Some(dim));
                    }
                    (array_shape[dim as usize], array_stride[dim as usize])
                } else {
                    over = true;
                    (0, 0)
                }
            })
            .collect::<(Vec<u32>, Vec<u32>)>();

        if over {
            let arr = format!(
                "Permute Error, {:?} Crosses The Array Boundary Of Array {:?}",
                permute, array_shape
            );
            return Err(ArrOgpuErr::Permute(arr));
        } else if repeat.0 {
            let arr = format!("Permute Error, Repeat Index Detected On {:?}", permute);
            return Err(ArrOgpuErr::Permute(arr));
        }

        let binding = self.create_metadata_binding(
            &array.pointer_to_arr(),
            &out_shape.0,
            &get_stride_from_shape(&out_shape.0),
            &out_shape.1,
            &array.offset(),
        );

        let shape_padding: [u32; 12] = vector_padding(out_shape.0.clone(), 0, 12)
            .map_err(|err| ArrOgpuErr::Permute(err))?
            .try_into()
            .unwrap();

        let origin_stride_padding: [u32; 12] =
            vector_padding(get_stride_from_shape(&out_shape.0), 0, 12)
                .map_err(|err| ArrOgpuErr::Permute(err))?
                .try_into()
                .unwrap();

        let stride_padding: [u32; 12] = vector_padding(out_shape.1.clone(), 0, 12)
            .map_err(|err| ArrOgpuErr::Permute(err))?
            .try_into()
            .unwrap();

        let metadata = self.create_metadata_compound(
            array.pointer_to_arr(),
            array.len(),
            array.dim() as u32,
            array.offset(),
            shape_padding,
            stride_padding,
            origin_stride_padding,
        );

        let array_view = GpuArrayView {
            array: array,
            offset: array.offset(),
            pointer: array.pointer(),
            shape: out_shape.0,
            stride: out_shape.1,
            binding: Some(binding),
            metadata_compound: Some(metadata),
        };

        Ok(array_view)
    }
}
