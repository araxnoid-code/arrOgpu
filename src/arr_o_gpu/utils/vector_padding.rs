pub fn vector_padding<T>(
    mut vector: Vec<T>,
    padding_value: T,
    padding: usize,
) -> Result<Vec<T>, String>
where
    T: Clone,
{
    let len = vector.len();

    if len > padding {
        return Err("PADDING ERROR, Vector Length Is Greater Than Target Padding".to_string());
    }
    vector.append(&mut vec![padding_value; padding - len]);
    Ok(vector)
}

// [1,2,3]
// 10
