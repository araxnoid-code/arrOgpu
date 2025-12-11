use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView };

impl ArrOgpuModule {
    pub fn matmul_view<'a, A, B>(&self, array_a: &A, array_b: &B) -> Result<(), ArrOgpuErr>
        where A: ArrayView, B: ArrayView
    {
        if array_a.dim() != 2 || array_b.dim() != 2 {
            let err = format!(
                "Array matmul 2d Error, dim of array A is {} and dim of array B is {}",
                array_a.dim(),
                array_b.dim()
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }

        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        let m_k = [shape_a[0], shape_a[1]];
        let k_n = [shape_b[0], shape_b[1]];
        if m_k[1] != k_n[0] {
            let err = format!(
                "Array matmul 2d Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a,
                shape_b
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }
        let out_shape = [m_k[0], k_n[1]];

        Ok(())
    }
}
