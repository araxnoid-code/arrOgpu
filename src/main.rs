use arr_o_gpu::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArray, r };

fn main() {
    let data = 10;
    get_data(&data);

    let data = "halo";
    get_data(data);

    let data = false;
    get_data(&data);
}

fn get_data<T>(data: &T) where T: ?Sized {}
