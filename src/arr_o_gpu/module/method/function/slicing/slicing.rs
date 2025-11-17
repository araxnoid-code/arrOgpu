use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray, SliceRange };

impl ArrOgpuModule {
    pub fn slicing(&self, array: &GpuArray, slice: &mut [SliceRange]) -> Result<(), ArrOgpuErr> {
        if array.shape().len() < slice.len() {
            let err = format!(
                "Array Slicing Error, Array {:?} can't Slice By {:?} cause out of range",
                array.shape(),
                slice
            );

            return Err(ArrOgpuErr::Slicing(err));
        }

        // check
        let mut error = None;
        for (i, range) in slice.iter_mut().enumerate() {
            let start = range.start.unwrap_or(0);
            let end = range.end.unwrap_or(array.shape()[i] as usize);

            println!("{}..{}", start, end);

            if start >= end || end > (array.shape()[i] as usize) {
                error = Some(range.clone());
                break;
            }

            if let None = range.start {
                range.start = Some(start);
            }

            if let None = range.end {
                range.end = Some(end);
            }
        }
        if let Some(range) = error {
            let err = format!(
                "Array Slicing Error, Error detected for {:?} in slice {:?}",
                range,
                slice
            );
            return Err(ArrOgpuErr::Slicing(err));
        }

        Ok(())
    }
}
