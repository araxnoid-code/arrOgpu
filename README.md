<div align="center">
    <img width="250px" src="./image/arrOgpu_logo.png"></img>
    <h1>arrOgpu</h1>
    <p>under development ⚙️</p>
</div>

Matrix Operations On GPU

## Base On WGPU
This library uses [`WGPU`](https://wgpu.rs/) to perform matrix operations

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



