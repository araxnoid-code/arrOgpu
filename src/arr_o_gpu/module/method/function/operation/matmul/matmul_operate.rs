use std::sync::RwLockWriteGuard;

use crate::{Allocator, ArrOgpuErr, ArrayCompute, get_stride_from_shape};

pub(crate) enum MatmulOperate<'a, A, B>
where
    A: ArrayCompute,
    B: ArrayCompute,
{
    _2D(&'a A, &'a B),
    ND(&'a A, &'a B),
}

impl<'a, A, B> MatmulOperate<'a, A, B>
where
    A: ArrayCompute,
    B: ArrayCompute,
{
    pub fn create(array_a: &'a A, array_b: &'a B) -> Result<MatmulOperate<'a, A, B>, ArrOgpuErr> {
        let dim = array_a.dim();
        if array_a.shape().len() <= 1 || array_b.shape().len() <= 1 {
            let err = format!(
                "Array Matmul 2d Error, Shape Of Array A Is {:?} And Shape Of Array B Is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            Err(ArrOgpuErr::Matmul2D(err))
        } else if array_a.shape().len() == 2
            && array_b.shape().len() == 2
            && array_a.shape()[1] == array_b.shape()[0]
        {
            Ok(Self::_2D(array_a, array_b))
        } else if array_a.shape().len() == array_b.shape().len()
            && array_a.shape()[..dim - 2] == array_b.shape()[..dim - 2]
            && array_a.shape()[dim - 1] == array_b.shape()[dim - 2]
        {
            Ok(Self::ND(array_a, array_b))
        } else {
            let err = format!(
                "Array Matmul 2d Error, Shape Of Array A Is {:?} And Shape Of Array B Is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            Err(ArrOgpuErr::Matmul2D(err))
        }
    }

    pub fn create_metadata_output(
        &self,
        allocate: &mut RwLockWriteGuard<'_, Allocator>,
    ) -> (
        Vec<u32>,
        u32,
        Vec<u32>,
        u32,
        u32,
        (crate::SpaceType, u32, u32),
    ) {
        match self {
            Self::_2D(arr_a, arr_b) => {
                let shape = vec![arr_a.shape()[0], arr_b.shape()[1]];
                let len = shape.iter().product::<u32>();
                let stride = get_stride_from_shape(&shape);
                let dim = 2;
                let offset = 0;
                let allocate = allocate.pointer_input(len);

                (shape, len, stride, dim, offset, allocate)
            }

            Self::ND(arr_a, arr_b) => {
                let mut shape = arr_a.shape().clone();
                if let Some(coll) = shape.last_mut() {
                    *coll = *arr_b.shape().last().unwrap();
                }
                let len = shape.iter().product::<u32>();
                let stride = get_stride_from_shape(&shape);
                let dim = arr_a.dim() as u32;
                let offset = 0;
                let allocate = allocate.pointer_input(len);

                (shape, len, stride, dim, offset, allocate)
            }
        }
    }

    pub fn get_x_y_z(&self) -> (u32, u32, u32) {
        match self {
            Self::_2D(arr_a, arr_b) => {
                let x = (arr_a.shape()[0] + 15) / 16;
                let y = (arr_b.shape()[1] + 15) / 16;
                (x, y, 1)
            }
            Self::ND(arr_a, arr_b) => {
                let x = (arr_a.shape()[arr_a.shape().len() - 2] + 15) / 16;
                let y = (arr_b.shape()[arr_b.shape().len() - 1] + 15) / 16;
                let z = arr_a.shape()[..arr_a.dim() - 2].iter().product::<u32>();
                (x, y, z)
            }
        }
    }
}
