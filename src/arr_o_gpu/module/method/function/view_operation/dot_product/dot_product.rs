use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView };

impl ArrOgpuModule {
    pub fn dot_product_view<A, B>(&self, array_a: A, array_b: B) -> Result<(), ArrOgpuErr>
        where A: ArrayView, B: ArrayView
    {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        let length_a = array_a.len();
        let length_b = array_b.len();

        if shape_a.len() != 1 || shape_b.len() != 1 || length_a != length_b {
            let err = format!(
                "Array Dot Product Error, Array with shape {:?} and With Shape {:?} Can't Be Operated",
                shape_a,
                shape_b
            );
            return Err(ArrOgpuErr::DotProduct(err));
        }

        Ok(())
    }
}
