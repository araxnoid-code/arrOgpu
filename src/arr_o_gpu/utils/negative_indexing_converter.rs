use crate::{ArrOgpuErr, SliceRange, SliceRangeNegativeAble};

impl From<String> for ArrOgpuErr {
    fn from(value: String) -> Self {
        Self::NegativeIndexing(value)
    }
}

pub fn negative_indexing_converter(index: &[i32], shape: &[u32]) -> Result<Vec<u32>, String> {
    let mut indexing: Vec<u32> = Vec::with_capacity(index.len());

    if index.len() > shape.len() {
        return Err(format!(
            "Negative Indexing Converter Error, Negative Indexing Dimension {:?} Overflow On Shape Dimension {:?}",
            index.len(),
            shape.len()
        ));
    }

    //
    for (i, index) in index.iter().enumerate() {
        if index < &0 {
            let idx = shape[i] as i32 + index;
            if idx < 0 {
                return Err(format!(
                    "Negative Indexing Converter Error, Negative Indexing {:?} Overflow On Shape {:?}",
                    index, shape
                ));
            }

            indexing.push(idx as u32);
        } else {
            indexing.push(*index as u32);
        }
    }
    //

    Ok(indexing)
}

pub fn slice_negative_indexing_converter(
    slice: &[SliceRangeNegativeAble],
    shape: &[u32],
) -> Result<Vec<SliceRange>, String> {
    if slice.len() > shape.len() {
        return Err(format!(
            "Negative Indexing Converter Error, Slice Negative Indexing Dimension {:?} Overflow On Shape Dimension {:?}",
            slice.len(),
            shape.len()
        ));
    }

    let mut indexing: Vec<SliceRange> = Vec::with_capacity(slice.len());

    for (i, slice_negative_able) in slice.iter().enumerate() {
        let dim = shape[i] as i32;
        let mut start = slice_negative_able.start.unwrap_or(0);
        let mut end = slice_negative_able.end.unwrap_or(dim);

        if start < 0 {
            start = dim + start;
        }

        if end < 0 {
            end = (dim + end) + 1;
        }

        if start == end {
            return Err(format!(
                "Negative Indexing Converter Error, Slicing With Range {:?} Will Return An Empty Array, Empty Array Not Supported",
                slice
            ));
        } else if start > end {
            return Err(format!(
                "Negative Indexing Converter Error, Slicers with range {:?} have a start greater than an end.",
                slice,
            ));
        }

        let slice_range = SliceRange {
            start: Some(start as u32),
            end: Some(end as u32),
        };

        indexing.push(slice_range);
    }

    Ok(indexing)
}

// 0..4
