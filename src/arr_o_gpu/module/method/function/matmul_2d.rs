use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray };

impl ArrOgpuModule {
    pub fn matmul_2d(&self, arr_a: GpuArray, arr_b: GpuArray) -> Result<(), ArrOgpuErr> {
        if arr_a.dim() != 2 || arr_b.dim() != 2 {
            let err = format!(
                "Array matmul 2d Error, dim of array A is {} and dim of array B is {}",
                arr_a.dim(),
                arr_b.dim()
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }

        Ok(())
    }
}
