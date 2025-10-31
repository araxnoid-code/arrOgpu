use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray };

impl ArrOgpuModule {
    pub fn matmul_nd(&self, array_a: &GpuArray, array_b: &GpuArray) -> Result<(), ArrOgpuErr> {
        let shape_a = array_a.shape();
        let k_a = shape_a[shape_a.len() - 1];
        let shape_b = array_b.shape();
        let k_b = shape_b[shape_b.len() - 2];

        if
            shape_a.len() != shape_b.len() ||
            &shape_a[0..shape_a.len() - 2] != &shape_b[0..shape_a.len() - 2] ||
            k_a != k_b
        {
            let err = format!(
                "Array matmul nd Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a,
                shape_b
            );

            return Err(ArrOgpuErr::MatmulND(err));
        }

        Ok(())
    }
}
