use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray };

impl ArrOgpuModule {
    pub fn broadcasting(&self, arr: &GpuArray, broadcast: &[u32]) -> Result<(), ArrOgpuErr> {
        let arr_shape = arr.shape();
        if arr_shape.len() > broadcast.len() {
            let err = format!(
                "Array Broadcasting Error, Array {:?} can't Broadcast to {:?}",
                arr_shape,
                broadcast
            );

            return Err(ArrOgpuErr::Broadcast(err));
        }

        // expend shape
        let diff_range = broadcast.len() - arr_shape.len();
        let extend_arr_shape = if diff_range != 0 {
            let mut extend = vec![1;diff_range];
            extend.extend_from_slice(arr_shape);
            extend
        } else {
            arr_shape.clone()
        };

        // validation every dim
        let mut broadcast_index = vec![];
        for i in (0..broadcast.len()).rev() {
            let arr_dim = extend_arr_shape[i];
            let broadcast_dim = broadcast[i];

            if arr_dim != broadcast_dim {
                if arr_dim == 1 {
                    broadcast_index.push(i);
                } else {
                    let err = format!(
                        "Array Broadcasting Error, Array {:?} can't Broadcast to {:?}",
                        arr_shape,
                        broadcast
                    );

                    return Err(ArrOgpuErr::Broadcast(err));
                }
            }
        }

        println!("{:?}", broadcast_index);

        Ok(())
    }
}
