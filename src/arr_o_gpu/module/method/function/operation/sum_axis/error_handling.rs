use std::collections::HashSet;

use crate::{ ArrOgpuErr, ArrayView };

pub(crate) fn error_handling<A>(array_a: &A, axis: &[u32]) -> Result<(), ArrOgpuErr>
    where A: ArrayView
{
    // indexing out of shape
    let shape = array_a.shape();
    let axis_len = axis.len();
    if axis_len > shape.len() {
        let error = format!(
            "Sum Axis Error, Indexing On {:?} Exceeds The Dimension Of Array {:?}",
            axis,
            array_a.shape()
        );
        return Err(ArrOgpuErr::SumAxis(error));
    } else if axis_len == 0 {
        let error = format!("Sum Axis Error, Axis Cannot Be Empty");
        return Err(ArrOgpuErr::SumAxis(error));
    }

    // repeated index & check all index on axis
    let mut value: HashSet<usize> = HashSet::new();
    let array_dim = shape.len();
    for &idx in axis {
        if (idx as usize) >= array_dim {
            let error = format!(
                "Sum Axis Error, The Index On {:?} Is Greater Than The Dimension On Array {:?}",
                axis,
                shape
            );
            return Err(ArrOgpuErr::SumAxis(error));
        }

        let idx = idx as usize;
        if let None = value.get(&idx) {
            value.insert(idx);
        } else {
            let error = format!("Sum Axis Error, Found Repeated Indexes On The Axis {:?}", axis);
            return Err(ArrOgpuErr::SumAxis(error));
        }
    }

    Ok(())
}
