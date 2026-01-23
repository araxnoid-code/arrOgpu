 <div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <b><p>Array Operations On Gpu</p></b>
    <p>⚙️ under development ⚙️</p>
    <b>
        <p>Version / 0.2.0</p>
    </b>
</div>

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform array operations

## What's New
Version / 0.2.0: [version.md](./version.md)

## Version Changelog
version changelog: [changelog.md](./changelog.md)

## Starting
### install
coming soon

### Code
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[1., 2., 3., 4., 5., 6.], &[2, 3])
        .unwrap();

    let result = array.add(&1.).unwrap();

    println!("{}", result);
    // [
    //  [2.0, 3.0, 4.0]
    //  [5.0, 6.0, 7.0]
    // ]
}
```

## List Method
- add
- sub
- mul
- div
- matmul
- sum
- sum_axis
- Sum_axis_keep_dim
- dot_product
- abs
- log2
- pow
- sin
- cos
- tan
- sqrt
- broadcasting
- slice
- index
- to_shape
- permute
- contiguous(View)
