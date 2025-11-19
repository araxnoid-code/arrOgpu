use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray, SliceRange, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn slicing(&self, array: &GpuArray, slice: &[SliceRange]) -> Result<(), ArrOgpuErr> {
        if array.shape().len() < slice.len() {
            let err = format!(
                "Array Slicing Error, Array {:?} can't Slice By {:?} cause out of range",
                array.shape(),
                slice
            );

            return Err(ArrOgpuErr::Slicing(err));
        }

        // check & get
        let wgpu_init = self.wgpu_init.read().unwrap();
        let mut allocator = self.allocator.write().unwrap();

        let mut output_shape = vec![];
        let output_allocate = allocator.pointer_input(output_shape.iter().product());
        let output_pointer = [output_allocate.1, output_allocate.2];

        let mut start_slice = vec![];
        let mut end_slice = vec![];

        let array_stride = array.stride();
        let array_pointer = array.pointer_to_arr();

        let output_stride = get_stride_from_shape(&output_shape);

        for (i, _) in array.shape().iter().enumerate() {
            if let Some(range) = slice.get(i) {
                let start = range.start.unwrap_or(0);
                let end = range.end.unwrap_or(array.shape()[i]);

                if start >= end || end > array.shape()[i] {
                    let err = format!(
                        "Array Slicing Error, Error detected for {:?} in slice {:?}",
                        range,
                        slice
                    );
                    return Err(ArrOgpuErr::Slicing(err));
                }

                start_slice.push(start);
                end_slice.push(end);
                output_shape.push((end - start) as u32);
            } else {
                output_shape.push(array.shape()[i] as u32);
            }
        }

        let mut _loop = vec![];
        for i in 0..output_shape.len() - 1 {
            let iter = output_shape[i + 1..output_shape.len() - 1].iter().product::<u32>();
            _loop.push(iter);
        }

        Ok(())
    }
}
