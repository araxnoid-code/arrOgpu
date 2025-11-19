<div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <b><p>Array Operations On Gpu</p></b>
    <p>⚙️ under development ⚙️</p>
    <p>version 0.0.0.5.7</p>
</div>

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform array operations

## 📦️ Installation
```toml
[dependencies]
arrOgpu = {git="https://github.com/araxnoid-code/arrOgpu.git"}
```

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
- in version 0.0.1 there are still a few features and the possibility of bugs will occur, in the future it will continue to be developed.

## Method
### Create An Array
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let model = ArrOgpuModule::default();

    let array = model
        .array_from_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 5])
        .unwrap();
    println!("{}", array)
    // output
    // [
    //  [1.0, 2.0, 3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0, 9.0, 10.0]
    // ]
}
```

### Indexing
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let model = ArrOgpuModule::default();

    let array = model
        .array_from_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 5])
        .unwrap();
    println!("{}", array);
    // [
    //  [1.0, 2.0, 3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0, 9.0, 10.0]
    // ]

    let indexing_a = array.index(&[0]).unwrap();
    println!("{}", indexing_a);
    // [1.0, 2.0, 3.0, 4.0, 5.0]

    let indexing_b = array.index(&[1, 4]).unwrap();
    println!("{}", indexing_b);
    // [10.0]
}
```

### element-wise operations(Add, Sub, Mul, Div)
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let model = ArrOgpuModule::default();

    let array_a = model
        .array_from_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 5])
        .unwrap();

    let array_b = model
        .array_from_vector(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], &[2, 5])
        .unwrap();

    let add = model.add(&array_a, &array_b).unwrap();
    println!("{}", add);
    // [
    //  [11.0, 13.0, 15.0, 17.0, 19.0]
    //  [21.0, 23.0, 25.0, 27.0, 29.0]
    // ]

    let sub = model.sub(&array_a, &array_b).unwrap();
    println!("{}", sub);
    // [
    //  [-9.0, -9.0, -9.0, -9.0, -9.0]
    //  [-9.0, -9.0, -9.0, -9.0, -9.0]
    // ]

    let mul = model.mul(&array_a, &array_b).unwrap();
    println!("{}", mul);
    // [
    //  [10.0, 22.0, 36.0, 52.0, 70.0]
    //  [90.0, 112.0, 136.0, 162.0, 190.0]
    // ]
    
    let div = model.div(&array_a, &array_b).unwrap();
    println!("{}", div);
    // [
    //  [0.1, 0.18181819, 0.25, 0.3076923, 0.35714284]
    //  [0.39999998, 0.4375, 0.47058824, 0.5, 0.5263158]
    // ]
}
```

### Matmul 2D
```rust
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let model = ArrOgpuModule::default();

    let array_a = model
        .array_from_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 5])
        .unwrap();

    let array_b = model
        .array_from_vector(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], &[5, 2])
        .unwrap();

    let result = model.matmul_2d(&array_a, &array_b).unwrap();
    println!("{}", result);
    // [
    //  [230.0, 245.0]
    //  [580.0, 620.0]
    // ]
}
```

### Matmul ND
```rs
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = module
        .array_from_vector(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], &[2, 2, 2])
        .unwrap();
    println!("{}", arr_a);

    // [
    //  [
    //   [1.0, 2.0]
    //   [3.0, 4.0]
    //  ]
    //  [
    //   [1.0, 2.0]
    //   [3.0, 4.0]
    //  ]
    // ]

    let arr_b = module
        .array_from_vector(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], &[2, 2, 2])
        .unwrap();
    println!("{}", arr_b);
    // [
    //  [
    //   [7.0, 8.0]
    //   [9.0, 10.0]
    //  ]
    //  [
    //   [11.0, 12.0]
    //   [13.0, 14.0]
    //  ]
    // ]

    let matmul = module.matmul_nd(&arr_a, &arr_b).unwrap();
    println!("{}", matmul);
    // [
    //  [
    //   [25.0, 28.0]
    //   [57.0, 64.0]
    //  ]
    //  [
    //   [37.0, 40.0]
    //   [85.0, 92.0]
    //  ]
    // ]
}
```

### Broadcasting
```rs
use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = module.array_from_vector(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]).unwrap();
    println!("{}", arr_a);
    // [
    //  [
    //   [1.0, 2.0]
    //  ]
    //  [
    //   [3.0, 4.0]
    //  ]
    // ]

    let broadcasting = module.broadcasting(&arr_a, &[2, 3, 2]).unwrap();
    println!("{}", broadcasting);
    // [
    //  [
    //   [1.0, 2.0]
    //   [1.0, 2.0]
    //   [1.0, 2.0]
    //  ]
    //  [
    //   [3.0, 4.0]
    //   [3.0, 4.0]
    //   [3.0, 4.0]
    //  ]
    // ]
}

```

### Slicing
```rust
use arr_o_gpu::{ ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = module
        .array_from_vector(
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0,
            ],
            &[2, 3, 3]
        )
        .unwrap();
    println!("{}", arr_a);
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
    // ]

    let slicing_a = module.slicing(&arr_a, &[r(1..2)]).unwrap();
    println!("{}", slicing_a);
    // [
    //  [
    //   [9.0, 10.0, 11.0]
    //   [12.0, 13.0, 14.0]
    //   [15.0, 16.0, 17.0]
    //  ]
    // ]

    let slicing_b = module.slicing(&arr_a, &[r(..), r(1..), r(..2)]).unwrap();
    println!("{}", slicing_b);
    // [
    //  [
    //   [3.0, 4.0]
    //   [6.0, 7.0]
    //  ]
    //  [
    //   [12.0, 13.0]
    //   [15.0, 16.0]
    //  ]
    // ]
}

```


