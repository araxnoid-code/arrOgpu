<div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <b><p>Array Operations On Gpu</p></b>
    <p>⚙️ under development ⚙️</p>
    <p>nightly / 0.1.0.1</p>
</div>

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform array operations

## 🚧 Nightly Version!
is an experimental version used in development and testing.

## Code
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let model = ArrOgpuModule::default();
}
```
### 🚧 Announcement
- arrOgpu only supports numbers of type f32.
- In this version, arrOgpu is still in the development stage, therefore the maximum storage allowed is 400000 bytes or 100000 numbers of type f32.
- in this version there are still a few features and the possibility of bugs will occur, in the future it will continue to be developed.

## What New?
### fix bug
- vector parameters data type bug in array_from_vector method

### Optimaze Code
- 2d matmul(wgsl file)
- nd matmul(wgsl file)
- Broadcasting(wgsl file)
- Slicing(wgsl file)

### feature
#### GpuArrayView
view mode for array
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module.array_from_vector(&[0.0, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
    let array_view = module.array_view_from_array(&array);
    // or
    let array_view = array.view();
}
```

`collect` method to get the full array view mode
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module.array_from_vector(&[0.0, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
    let array_view = module.array_view_from_array(&array);
    // or
    let array_view = array.view();

    let another_array = array_view.collect();
}
```

#### index_view()
still `bug`

indexing on an array that will return a GpuArrayView
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module.array_from_vector(&[0.0, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
    let indexing = module.index_view(&array, &[0]).unwrap();
}
```



