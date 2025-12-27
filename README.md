<div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <b><p>Array Operations On Gpu</p></b>
    <p>⚙️ under development ⚙️</p>
    <b>
        <p>nightly / 0.1.0.4</p>
    </b>
</div>

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform array operations

## Nightly Version!
is an experimental version used in development and testing.

📜 Full changelog: [version.md](./version.md)

## Code
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();
}
```
### Announcement
- arrOgpu only supports numbers of type f32.
- in this version there are still a few features and the possibility of bugs will occur, in the future it will continue to be developed.

## What New?
### Optimize
  - Rewrite Sum Compute Shader (wgsl file) to optimize using as many threads as possible.

### Change
  - Change Trait ArrayView Name to ArrayCompute
  - Change reshape Method to to_shape

### System
  - Added CheckArrayType Trait to check if array is contiguous or view.
  - Trait ContiguousArray and Trait ViewArray (for separate GpuArray and GpuArrayView)
  - summarizes all operational methods of arrays
  ```rust
use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();

    let something = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();

    array.add(&something).unwrap(); // == module.add(&array, &something);
    array.sin().unwrap();           // == module.sin(&array);
    // etc
}

  ```

### Features
#### abs, sqrt, log2, pow
```rust
use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();
    println!("{}", array);

    let abs = array.abs().unwrap();

    let sqrt = array.sqrt().unwrap();

    let log2 = array.log2().unwrap();

    let pow = array.pow(&2.).unwrap();
    // or
    let scalar = module.array_from_vector(&[2.0], &[1]).unwrap();
    let pow = array.pow(&scalar).unwrap();
}
```
#### Negative Indexing:
index method
```rust
use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();
    println!("{}", array);
    // [
    // [0.0, 1.0, 2.0]
    // [3.0, 4.0, 5.0]
    // [6.0, 7.0, 8.0]
    // [9.0, 10.0, 11.0]
    // ]

    let index = array.index(&[-1]).unwrap(); // == array.index(&[3]);
    println!("{}", index.contiguous());
    // [9.0, 10.0, 11.0]

    let index = array.index(&[-2, -3]).unwrap(); // == array.index(&[2, 0]);
    println!("{}", index.contiguous());
    // [6.0]
}
```

slicing method
```rust
use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();
    println!("{}", array);
    // [
    // [0.0, 1.0, 2.0]
    // [3.0, 4.0, 5.0]
    // [6.0, 7.0, 8.0]
    // [9.0, 10.0, 11.0]
    // ]

    let slice = array.slicing(&[r(..-1)]).unwrap(); // == array.slicing(&[0..4]);
    println!("{}", slice.contiguous());
    // [
    // [0.0, 1.0, 2.0]
    // [3.0, 4.0, 5.0]
    // [6.0, 7.0, 8.0]
    // [9.0, 10.0, 11.0]
    // ]

    let slice = array.slicing(&[r(..-2), r(1..-2)]).unwrap(); // == array.slicing(&[0..3, 1..2]);
    println!("{}", slice.contiguous());
    // [
    // [1.0]
    // [4.0]
    // [7.0]
    // ]
}
```
