# VERSION/0.2.1
- memory allocator now uses monagement, [monagement](https://github.com/araxnoid-code/monagement)
- feature to set limitations
  - change limitations based on driver capabilities,check:
```rust
use arr_o_gpu::ArrOgpuModule;
fn main() {
    let module = ArrOgpuModule::default();
    let driver_limit = module.get_adapter_limit();// driver limit
    let device_limit = module.get_device_limit();// limits used by wgpu
}
```
- execute_opt feature,provides the option to set workgroup_size.The operation that can do this currently:
  - add
  - sub
  - mul
  - div
  - matmul
    - matmul_2d
      - note:using 2 dimensions.if set to 16 then 16x16=256
    - matmul_nd
      - note:similar to matmul_2d,but dim z cannot be set yet
- The use of execute_opt depends on the limitations.The maximum storage can also be increased by changing the limitations,for example:
```rust
use arr_o_gpu::{AddOpt,ArrOgpuModule,FunctionExecuteOpt,HeapSize,WgpuLimits};
fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        limits: WgpuLimits {
            max_storage_buffer_binding_size: 256 << 20,// 256MiB
            max_compute_workgroup_size_x: 512,
            ..Default::default()
        },
        function_execute_opt: FunctionExecuteOpt {
            add: AddOpt {
                compute_workgroup_size_x: 512,// max_compute_workgroup_size_x must be 512
            },
            ..Default::default()
        },
        heap_size: HeapSize::Item(39321600),// 150MiB
        ..Default::default()
    });
}
```
- expanding max_storage_buffer_binding_size,must pay attention to max_buffer_size
```rust
use arr_o_gpu::{ArrOgpuModule,WgpuLimits};
fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        limits: WgpuLimits {
            max_storage_buffer_binding_size: 512 << 20,
            max_buffer_size: 512 << 20,
            ..Default::default()
        },
        ..Default::default()
    });
}
```
- fixed a bug in matmul due to the difference in override names on the wgsl and rust sides
- resolved the matmul_nd_view.wgsl bug, in the form of tile_b not being updated
- resolved the matmul_nd_view.wgsl bug, which was a logic error in mapping the location of the value array from metadata
- add testing code
