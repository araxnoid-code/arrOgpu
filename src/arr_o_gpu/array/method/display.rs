use std::fmt::Display;

use crate::GpuArray;

impl Display for GpuArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        let shape = &self.shape;
        let vector = self.get_heap();
        recursion(&mut output, shape, &vector, 0, &mut "".to_string());

        f.write_str(&output)
    }
}

fn recursion(output: &mut String, shape: &[u32], vector: &[f32], stride: u32, space: &mut String) {
    if shape.len() > 1 {
        output.push_str(&format!("{}[\n", space));
        space.push_str(" ");
        for i in 0..shape[0] {
            let stride = stride + i * shape[1..].iter().product::<u32>();
            recursion(output, &shape[1..], vector, stride, space);
        }

        space.pop();
        output.push_str(&format!("{}]\n", space));
    } else {
        // row / dim = 1
        let len = shape[0];
        let start = stride as usize;
        let end = start + (len as usize);
        let row = format!("{}{:?}\n", space, &vector[start..end]);
        output.push_str(&row);
    }
}
