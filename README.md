<div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <b><p>Array Operations On Gpu</p></b>
    <p>⚙️ under development ⚙️</p>
    <b>
        <p>nightly / 0.1.0.3</p>
    </b>
</div>

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform array operations

## 🚧 Nightly Version!
is an experimental version used in development and testing.

📜 Full changelog: [version.md](./version.md)

## Code
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();
}
```
### 🚧 Announcement
- arrOgpu only supports numbers of type f32.
- in this version there are still a few features and the possibility of bugs will occur, in the future it will continue to be developed.

## What New?
### Fixed Bug
- Fixed bug in matmul_nd method.

### System
- The maximum storage or heap capacity of ArrOgpuModule now does not have to be 100000 float numbers or 400000 bytes.
- Make the data type that implements the ArrayView trait(GpuArray and GpuArrayView) the primary data type in the parameters for the methods in the module.
- Remove the binding_compounds property from ArrOgpuModule, then add the heap_binding property to access the heap bind group.

### Optimaze
- Remove unnecessary poll.
- Remove no_longer_use folder.

### Features
#### Manual initialization of ArrOgpuModule.
```rust
use arr_o_gpu::{ ArrOgpuModule, ArrOgpuModuleInit };
fn main() {
    let module = ArrOgpuModule::default();
    // or
    let module = ArrOgpuModule::init(ArrOgpuModuleInit::default()).unwrap();
}
```
by default will set:
- The maximum storage capacity is 100000 float numbers or 400000 bytes.
- memory: MemoryUsage,
- power : LowPower,

to be more specific
```rust
use arr_o_gpu::{ ArrOgpuModule, ArrOgpuModuleInit, HeapSize, ManualInit, Memory, Power, WgpuInit };
fn main() {
    let module = ArrOgpuModule::init(ArrOgpuModuleInit {
        heap_size: HeapSize::Item(100000), // 100000 float numbers
        wgpu: WgpuInit::ManualInit(ManualInit {
            memory: Memory::Performance,
            power: Power::HighPerformance,
        }),
    }).unwrap();
}
```

further explanation:
```rust
pub struct ArrOgpuModuleInit {
    pub heap_size: HeapSize,
    pub wgpu: WgpuInit,
}
```

`heap_size` functions to set the maximum capacity of the heap.
```rust
pub enum HeapSize {
    Item(u32), // ex: 100 float numbers equals 400 bytes.
    Byte(u64), // ex: 800 bytes is equal to 200 float numbers.
}
```

`wgpu` serves to initialize wgpu.
```rust
pub enum WgpuInit {
    ManualDeviceQueue(Device, Queue), // Deeply create devices and queue
    ManualInit(ManualInit),
}

pub struct ManualInit {
    pub power: Power,
    pub memory: Memory,
}

pub enum Power {
    HighPerformance,
    LowPower,
}

pub enum Memory {
    Performance,
    MemoryUsage,
}
```


#### Allow Scalar In Element-Wise Operation.
##### add
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

    let add = module.add(&array, &10.0).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [10.0, 11.0, 12.0]
    //   [13.0, 14.0, 15.0]
    //  ]
    //  [
    //   [16.0, 17.0, 18.0]
    //   [19.0, 20.0, 21.0]
    //  ]
    // ]

    // or
    let scalar = module.array_from_vector(&[5.0], &[1]).unwrap();
    println!("{}", scalar);
    // [5.0]

    let add = module.add(&array, &scalar).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [5.0, 6.0, 7.0]
    //   [8.0, 9.0, 10.0]
    //  ]
    //  [
    //   [11.0, 12.0, 13.0]
    //   [14.0, 15.0, 16.0]
    //  ]
    // ]
}
```
##### sub
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

    let add = module.sub(&array, &10.0).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [-10.0, -9.0, -8.0]
    //   [-7.0, -6.0, -5.0]
    //  ]
    //  [
    //   [-4.0, -3.0, -2.0]
    //   [-1.0, 0.0, 1.0]
    //  ]
    // ]

    // or
    let scalar = module.array_from_vector(&[5.0], &[1]).unwrap();
    println!("{}", scalar);
    // [5.0]

    let add = module.sub(&array, &scalar).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [-5.0, -4.0, -3.0]
    //   [-2.0, -1.0, 0.0]
    //  ]
    //  [
    //   [1.0, 2.0, 3.0]
    //   [4.0, 5.0, 6.0]
    //  ]
    // ]
}
```
##### mul
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

    let add = module.mul(&array, &10.0).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [0.0, 10.0, 20.0]
    //   [30.0, 40.0, 50.0]
    //  ]
    //  [
    //   [60.0, 70.0, 80.0]
    //   [90.0, 100.0, 110.0]
    //  ]
    // ]

    // or
    let scalar = module.array_from_vector(&[5.0], &[1]).unwrap();
    println!("{}", scalar);
    // [5.0]

    let add = module.mul(&array, &scalar).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [0.0, 5.0, 10.0]
    //   [15.0, 20.0, 25.0]
    //  ]
    //  [
    //   [30.0, 35.0, 40.0]
    //   [45.0, 50.0, 55.0]
    //  ]
    // ]
}
```
##### div
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

    let add = module.div(&array, &10.0).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [0.0, 0.1, 0.2]
    //   [0.3, 0.4, 0.5]
    //  ]
    //  [
    //   [0.6, 0.7, 0.8]
    //   [0.90000004, 1.0, 1.1]
    //  ]
    // ]

    // or
    let scalar = module.array_from_vector(&[5.0], &[1]).unwrap();
    println!("{}", scalar);
    // [5.0]

    let add = module.div(&array, &scalar).unwrap();
    println!("{}", add);
    // [
    //  [
    //   [0.0, 0.2, 0.4]
    //   [0.6, 0.8, 1.0]
    //  ]
    //  [
    //   [1.2, 1.4, 1.6]
    //   [1.8000001, 2.0, 2.2]
    //  ]
    // ]
}
```

#### Sum Method.
##### sum
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

    let sum = module.sum(&array).unwrap();
    println!("{}", sum);
    // [66.0]
}
```

##### sum_axis
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

    let sum = module.sum_axis(&array, &[0]).unwrap();
    println!("{}", sum);
    // [
    //  [6.0, 8.0, 10.0]
    //  [12.0, 14.0, 16.0]
    // ]

    let sum = module.sum_axis(&array, &[0, 1]).unwrap();
    println!("{}", sum);
    // [18.0, 22.0, 26.0]
}
```

##### sum_axis_keep_dim
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

    let sum = module.sum_axis_keep_dim(&array, &[0]).unwrap();
    println!("{}", sum);
    // [
    //  [
    //   [6.0, 8.0, 10.0]
    //   [12.0, 14.0, 16.0]
    //  ]
    // ]

    let sum = module.sum_axis_keep_dim(&array, &[0, 1]).unwrap();
    println!("{}", sum);
    // [
    //  [
    //   [18.0, 22.0, 26.0]
    //  ]
    // ]
}
```

#### sin, cos, tan method.
```rust
use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };
fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 2, 3], &module)
        .unwrap();
    let sin = module.sin(&array).unwrap();
    let cos = module.cos(&array).unwrap();
    let tan = module.tan(&array).unwrap();
}
```
