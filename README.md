<div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <b><p>Array Operations On Gpu</p></b>
    <p>⚙️ under development ⚙️</p>
    <b>
        <p>nightly / 0.1.0.2</p>
    </b>
</div>

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform array operations

## 🚧 Nightly Version!
is an experimental version used in development and testing.

📜 Full changelog: [version.md](https://github.com/araxnoid-code/arrOgpu/blob/nightly/0.1.0.2/version.md)

## Code
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();
}
```
### 🚧 Announcement
- arrOgpu only supports numbers of type f32.
- In this version, arrOgpu is still in the development stage, therefore the maximum storage allowed is 400000 bytes or 100000 numbers of type f32.
- in this version there are still a few features and the possibility of bugs will occur, in the future it will continue to be developed.

## What New?
### Fixed Bug
- Fixed bug in index_view method.
- Fixed broadcasting_view not syncing with permute method due to stride.
- Fixed bug in matmul_2d.
- Fixed a bug in allocator management due to a logic error.

### Change
- change the name of collect method to contiguous method.

### System
- use of generic structures for all methods that allow operating GpuArrayView.
- Each GpuArray and GpuArrayView has a memory binding that stores the meta data of the array itself.

### Features
#### slicing_view method
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };
fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..27)
        .to_GpuArray_with_shape(&[3, 3, 3], &module)
        .unwrap();
    println!("{}", array);
    // [
    //  [
    //   [0.0, 1.0, 2.0]
    //   [3.0, 4.0, 5.0]
    //   [6.0, 7.0, 8.0]
    //  ]
    //  [
    //   [9.0, 10.0, 11.0]
    //   [12.0, 13.0, 14.0]
    //   [15.0, 16.0, 17.0]
    //  ]
    //  [
    //   [18.0, 19.0, 20.0]
    //   [21.0, 22.0, 23.0]
    //   [24.0, 25.0, 26.0]
    //  ]
    // ]

    let array_view = module.slicing_view(&array, &[r(..), r(0..2), r(1..3)]).unwrap();
    println!("{}", array_view.contiguous());
    // [
    //  [
    //   [1.0, 2.0]
    //   [4.0, 5.0]
    //  ]
    //  [
    //   [10.0, 11.0]
    //   [13.0, 14.0]
    //  ]
    //  [
    //   [19.0, 20.0]
    //   [22.0, 23.0]
    //  ]
    // ]
}
```

#### Broadcast_view method
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 1, 3], &module)
        .unwrap();
    println!("{}", array);
    // [
    //  [
    //   [0.0, 1.0, 2.0]
    //  ]
    //  [
    //   [3.0, 4.0, 5.0]
    //  ]
    // ]

    let array_view = module.broadcast_view(&array, &[2, 3, 3]).unwrap();
    println!("{}", array_view.contiguous());
    // [
    //  [
    //   [0.0, 1.0, 2.0]
    //   [0.0, 1.0, 2.0]
    //   [0.0, 1.0, 2.0]
    //  ]
    //  [
    //   [3.0, 4.0, 5.0]
    //   [3.0, 4.0, 5.0]
    //   [3.0, 4.0, 5.0]
    //  ]
    // ]
}
```

#### matmul_2d_view
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();
    println!("{}", array_a);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0]
    //  [9.0, 10.0, 11.0]
    // ]

    let array_b = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[3, 2], &module)
        .unwrap();
    println!("{}", array_b);
    // [
    //  [0.0, 1.0]
    //  [2.0, 3.0]
    //  [4.0, 5.0]
    // ]

    let array = module.matmul_2d_view(&array_a, &array_b).unwrap();
    println!("{}", array);
    // [
    //  [10.0, 13.0]
    //  [28.0, 40.0]
    //  [46.0, 67.0]
    //  [64.0, 94.0]
    // ]
}
```

#### matmul_nd_view
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 3, 2], &module)
        .unwrap();
    println!("{}", array_a);
    // [
    //  [
    //   [0.0, 1.0]
    //   [2.0, 3.0]
    //   [4.0, 5.0]
    //  ]
    //  [
    //   [6.0, 7.0]
    //   [8.0, 9.0]
    //   [10.0, 11.0]
    //  ]
    // ]

    let array_b = ArangeArray::arange(0..8)
        .to_GpuArray_with_shape(&[2, 2, 2], &module)
        .unwrap();
    println!("{}", array_b);
    // [
    //  [
    //   [0.0, 1.0]
    //   [2.0, 3.0]
    //  ]
    //  [
    //   [4.0, 5.0]
    //   [6.0, 7.0]
    //  ]
    // ]

    let array = module.matmul_nd_view(&array_a, &array_b).unwrap();
    println!("{}", array);
    // [
    //  [
    //   [2.0, 3.0]
    //   [6.0, 11.0]
    //   [10.0, 19.0]
    //  ]
    //  [
    //   [66.0, 79.0]
    //   [86.0, 103.0]
    //   [106.0, 127.0]
    //  ]
    // ]
}
```

#### dot_product_view
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..20)
        .to_GpuArray_with_shape(&[20], &module)
        .unwrap();
    println!("{}", array_a);
    // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]

    let array_b = ArangeArray::arange(20..40)
        .to_GpuArray_with_shape(&[20], &module)
        .unwrap();
    println!("{}", array_b);
    // [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]

    let array = module.dot_product_view(&array_a, &array_b).unwrap();
    println!("{}", array);
    // [6270.0]
}
```

#### reshape method
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..30)
        .to_GpuArray_with_shape(&[5, 6], &module)
        .unwrap();
    println!("{}", array);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    //  [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    //  [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    //  [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    // ]

    let array_view = module.reshape(&array, &[5, 2, 3]).unwrap();
    println!("{}", array_view.contiguous());
    // [
    //  [
    //   [0.0, 1.0, 2.0]
    //   [3.0, 4.0, 5.0]
    //  ]
    //  [
    //   [6.0, 7.0, 8.0]
    //   [9.0, 10.0, 11.0]
    //  ]
    //  [
    //   [12.0, 13.0, 14.0]
    //   [15.0, 16.0, 17.0]
    //  ]
    //  [
    //   [18.0, 19.0, 20.0]
    //   [21.0, 22.0, 23.0]
    //  ]
    //  [
    //   [24.0, 25.0, 26.0]
    //   [27.0, 28.0, 29.0]
    //  ]
    // ]
}
```

#### permute method
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 2, 3], &module)
        .unwrap();
    println!("{}", array);
    // [
    //  [
    //   [0.0, 1.0, 2.0]
    //   [3.0, 4.0, 5.0]
    //  ]
    //  [
    //   [6.0, 7.0, 8.0]
    //   [9.0, 10.0, 11.0]
    //  ]
    // ]

    let array_view = module.permute(&array, &[2, 0, 1]).unwrap();
    println!("{}", array_view.contiguous());
    // [
    //  [
    //   [0.0, 3.0]
    //   [6.0, 9.0]
    //  ]
    //  [
    //   [1.0, 4.0]
    //   [7.0, 10.0]
    //  ]
    //  [
    //   [2.0, 5.0]
    //   [8.0, 11.0]
    //  ]
    // ]
}
```

#### is_contiguous() method in ArrayView
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayView, r };
fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();
    println!("is_contiguous : {}", array.is_contiguous());
    println!("{}", array);
    // is_contiguous : true
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]

    let array_view = module.slicing_view(&array, &[r(..), r(..2)]).unwrap();
    println!("is_contiguous : {}", array_view.is_contiguous());
    println!("{}", array_view.contiguous());
    // is_contiguous : false
    // [
    //  [0.0, 1.0]
    //  [3.0, 4.0]
    // ]
}
```

#### get_heap method for arrayView will refer to the main array heap.
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();
    println!("{:?}", array.get_heap());
    // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    let array_view = module.slicing_view(&array, &[r(..), r(..2)]).unwrap();
    println!("{:?}", array_view.get_heap());
    // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
}
```

#### element wise operation which allow GpuArrayView
- add_view
- mul_view
- sub_view
- div_view

```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();

    module.add_view(&array_a, &array_b);
    module.sub_view(&array_a, &array_b);
    module.mul_view(&array_a, &array_b);
    module.div_view(&array_a, &array_b);
}
```