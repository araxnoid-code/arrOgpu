## Initialisasi
### menggunkana `ArrOgpuModule::default()`
menggunakan method default akan memberikan settingan default yang akan sesuai dengan settingan standart WGPU
```rust
use arr_o_gpu::ArrOgpuModule;
fn main() {
    let module = ArrOgpuModule::default();
}
```
### menggunakan init `ArrOgpuModule::init(ArrOgpuModuleInit)`
menggunakan method ini dalam initialisasi akan memberikan lebih banyak keluasan.
```rust
use arr_o_gpu::{
    ArrOgpuModule, ArrOgpuModuleInit, FunctionExecuteOpt, HeapSize, ManualInit, Memory, WgpuInit,
    WgpuLimits,
};

fn main() {
    let module = ArrOgpuModule::init(ArrOgpuModuleInit {
        heap_size: HeapSize::Item(10_000),
        limits: WgpuLimits {
            ..Default::default()
        },
        function_execute_opt: FunctionExecuteOpt::default(),
        wgpu: WgpuInit::ManualInit(ManualInit {
            memory: Memory::MemoryUsage,
            power: arr_o_gpu::PowerInit::LowPower,
        }),
    });
}
```
penjelasan ArrOgpuModuleInit:
- `heap_size` menerima enum `HeapSize`:
```rust
pub enum HeapSize {
    Item(u32),
    Byte(u64),
}
```
`heap_size` berfungsi untuk mengatur ukuran heap yang akan menjadi storage(memory pool) pada arrOgpu. arrOgpu secara standart akan menggunakan float 32bit.
`HeapSize::Item(u32)` berguna jika ingin menetapkan ukuran heap dengan satuan f32. contoh: `HeapSize::Item(1000)` akan mengatur heap yang dapat menampung 1000 float dan memiliki ukuran 4000 byte. `HeapSize::Byte(u64)` berguna saat ingin menetapkan ukuran heap dengan satuan byte, contouhnya saat `HeapSize::Byte(4000)` maka akan menampung sebesar 4000byte dengan kata lain 1000 float 32bit.

- `limits` berguna dalam mentapkan limit untuk WGPU secara costum, menerima struct `WgpuLimits`
```rust
pub struct WgpuLimits {
    pub max_storage_buffer_binding_size: u32,
    pub max_buffer_size: u64,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
}
```
berguna untuk fitur `execute_opt` nantinya.

- `function_execute_opt`, berfungsi untuk memberikan opsi dalam compile dan eksekusi shaders, masih beberapa fitur yang kompatibel dengan fitur ini, mengatur `function_execute_opt` harus sesuai dengan `limits` yang sudah dibahas tadi.
```rust
pub struct FunctionExecuteOpt {
    pub add: AddOpt,
    pub sub: SubOpt,
    pub mul: MulOpt,
    pub div: DivOpt,
    pub matmul: MatmulOpt,
}
```
- `wgpu`, berfungsi dalam initialisasi WGPU, menerima enum `WgpuInit`:
```rust
pub enum WgpuInit {
    ManualDeviceQueue(Adapter, Device, Queue),
    ManualInit(ManualInit),
}
```
`ManualDeviceQueue`, berfungsi di saat ingin menginisialisasikan WGPU secara manual atau ingin menggunakan WGPU yang sudah ada sebelumnya.

`ManualInit(ManualInit)` berfungsi untuk secara otomatis menginisialisasikan WGPU berdasarkan opsi yang bisa di atur melalui `ManualInit`:
```rust
pub struct ManualInit {
    pub power: PowerInit,
    pub memory: Memory,
}
```

### contoh penggunaan beberapa fungsi
```rust
use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    // membuat array
    // // note, type data yang didukung adalah f32 serta ukuran dimensi maksimal adala 8 dimensi
    // // menggunakan ArrOgpuModule::array_from_vector(&self, &[f32], &[u32])
    let array_a = module
        .array_from_vector(&[20., 30., 40., 50., 60., 70.], &[3, 2])
        .unwrap();
    // // cara lain dengan menggunakan Arange
    let array_b = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[3, 2], &module)
        .unwrap();

    // cetak array
    println!("{}", array_a);
    println!("{}", array_b);
    // note: setiap mencetak element akan mengakibatkan sinkronisasi antar cpu dan gpu yang meningkatkan latensi program
    // bijak dalam menggunakannya

    // operasi element wise sederhana
    // // penjumlahan
    let result = module.add(&array_a, &array_b).unwrap();
    println!("{}", result);
    // // pengurangan
    let result = module.sub(&array_a, &array_b).unwrap();
    println!("{}", result);
    // // perkalian
    let result = module.mul(&array_a, &array_b).unwrap();
    println!("{}", result);
    // // pembagian
    let result = module.div(&array_a, &array_b).unwrap();
    println!("{}", result);

    // mengambil data dari gpu
    let result = module.add(&array_a, &array_b).unwrap();
    let data = result.get_heap();
    println!("{:?}", data);

    // dot-product
    let array_a = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[6], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[6], &module)
        .unwrap();

    let result = module.dot_product(&array_a, &array_b).unwrap();
    println!("{}", result);

    // perkalian matrix
    let array_a = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[3, 2], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b).unwrap();
    println!("{}", result);
}
```

### Contiguous dan View
contiguous adalah array yang saling berdampingan dan continue di dalam heap, akan membuat proses pembacaan dan penulisan lebih cepat.
View adalah Array yang melompat-lompat cara membacanya, proses pembacaan akan memerlukan waktu namun sangat menghemat memory.
```rust
use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    // membuat array contiguous
    let array = ArangeArray::arange(0..128)
        .to_GpuArray_with_shape(&[8, 4, 4], &module)
        .unwrap();

    // merubah keseluruhan array menjadi view
    let array_view = module.array_view_from_array(&array);

    // fungsi yang mengakibatkan transform menjadi View
    let index = module.index(&array, &[1]).unwrap();
    // mendapatkan array berbentuk 4x4 berdasarkan array 8x4x4 sebelumnya
    // namun tidak mengalokasikan data baru karena View hanya merubah cara membaca data pada array 8x4x4 sebelumnya
    // untuk mencetaknya, harus diubah menjadi contiguos kembali
    let contiguous = index.contiguous().unwrap();
    println!("{}", contiguous);
}
```
array view masih dapat digunakan dalam operasi lainnya seperti matmul, dot-product, sum dan lain lain namun dengan compute shader yang berbeda dengan contiguous dan dengan peforma yang lebih lambat namun tidak ada pengalokasian memory sama sekali jadi gratis dalam biaya heap storage.

fungsi fungsi yang merubah array ke view
- index
- broadcast
- permute
- slicing
- to_shape
