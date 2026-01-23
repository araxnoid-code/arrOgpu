# VERSION/0.2.0
- attention
  - Although arrOgpu currently does not directly mention limitations, because arrOgpu is based on WGPU which has inherent limitations, therefore arrOgpu still has limitations on the maximum data that can be stored, how much data can be sent etc. therefore arrOgpu is still not ready for large cases, you can read about it at: https://docs.rs/wgpu/latest/wgpu/struct.Limits.html

- add cache
  - Now group 0 has 3 bindings, including
    - binding(0), called `heap`. Used for heap (storing numbers) with type var<storage, read_write>.
    - binding(1), called `execute_args`. It is a 256 bytes * 3, used to store metadata for arrays to be executed with type var<uniform>.
    - binding(2), called `static_cache`. It is a cache that stores static values with a size of 32 bytes with the data type var<uniform>.
    - added `pipeline_cache` to `arrOgpuModule` to cache used pipelines for reuse in subsequent executions.
    - added `common_pipeline_layout` in `arrOgpuModule`, which is a common pipeline_layout used by most compute shaders.
    - added the `PipelineCompound` enum which is used in many methods that work as tools for `pipeline_cache`.
  
- metadata update
  - Now, each array will store only a metadata buffer instead of a bind group. This metadata buffer will be passed to `execute_args`.
  Due to the metadata changes, some adjustments have been made, including:
    - added `metadata_compound: Option<MetadataCompound>,` property to GpuArray struct.
    - added `create metadata_compound` method to `arrOgpu Module` to create and add metadata buffers to be stored by the array.
    - The `binding` property on the GpuArray struct has its type changed to `Option<(BindGroupLayout, BindGroup)>,` which will be deprecated going forward.
    - added a `check_contiguous_or_view` method to the ArrayCompute trait to make it easier to select compute shaders.
    - The maximum array dimension is now 8 dimensions.

- rewrite compute shaders
  - The compute shaders used in execution have all been rewritten and optimized due to the use of new metadata. Compute shaders have now been combined into the compute_shaders folder for easier searching and maintenance, as can be seen in[compute_shaders](./src/arr_o_gpu/compute_shaders/)

- Changes to the method
  - pow has now been split into powi and powf.
  - matmul_nd and matmul_2d have now been merged into matmul, the matmul method itself will automatically determine whether to use matmul nd or matmul 2d compute shaders.
  - The data type bug in the slice method in arrOgpuModule which initially used `SliceRange` has now been changed to `SliceRangeNegativeAble`.

- reduction heap
  - In methods that use parallel reduction (sum, dot_product, sum_axis, sum_axis_keep_dim), additional memory is still used to store the reduction results between workgroups, which in the future will be stored in the main heap if possible.


# NIGHTLY/0.1.0.4
- Optimize
  - Rewrite Sum Compute Shader (wgsl file) to optimize using as many threads as possible.

- Change
  - Change Trait ArrayView Name to ArrayCompute
  - Change reshape Method to to_shape

- System
  - Added CheckArrayType Trait to check if array is contiguous or view.
  - Trait ContiguousArray and Trait ViewArray (for separate GpuArray and GpuArrayView)
  - summarizes all operational methods of arrays

- Features
  - abs
  - sqrt
  - log2
  - pow
  - Negative Indexing:
    - index method
    - slicing method

# NIGHTLY/0.1.0.3
- Fixed Bug
    - Fixed bug in matmul_nd method.
- System
    - The maximum storage or heap capacity of ArrOgpuModule now does not have to be 100000 float numbers or 400000 bytes.
    - Make the data type that implements the ArrayView(GpuArray and GpuArrayView) trait the primary data type in the parameters for the methods in the module.
    - Remove the binding_compounds property from ArrOgpuModule, then add the heap_binding property to access the heap binding group.
- Optimaze
    - Remove unnecessary poll.
    - Remove no_longer_use folder.
- Features
    - Manual initialization of ArrOgpuModule.
    - Allow Scalar In Element-Wise Operation.
        - add
        - sub
        - mul
        - div
    - Sum Method.
        - sum
        - sum_axis
        - sum_axis_keep_dim
    - sin, cos, tan method.

# NIGHTLY/0.1.0.2
- Fixed Bug
    - Fixed bug in index_view method.
    - Fixed broadcasting_view not syncing with permute method due to stride.
    - Fixed bug in matmul_2d.
    - Fixed a bug in allocator management due to a logic error.
- System
    - use of generic structures for all methods that allow operating GpuArrayView.
    - Each GpuArray and GpuArrayView has a memory binding that stores the meta data of the array itself.
- Features
    - slicing_view method
    - Broadcast_view method
    - matmul_2d_view
    - matmul_nd_view
    - dot_product_view
    - reshape method
    - permute method
    - is_contiguous() method in ArrayView
    - get_heap method for arrayView will refer to the main array heap.
    - element wise operation which allow GpuArrayView
        - add_view
        - mul_view
        - sub_view
        - div_view
- change
    - change the name of collect method to contiguous method

# NIGHTLY/0.1.0.1
- Bug Fix
    - parameter data type error in method array_from_vector

- Optimize
    - update matmul 2d boundary logic
    - update matmul nd boundary logic
    - minor changes to the broadcasting algorithm
    - minor changes to the slicing algorithm

- Features
    - GpuArrayView
    - collect for GpuArrayView (fitur)
    - index_view (fitur)

# Version/0.1.0
- new features
    - Maximum storage is 100000 item of float
    - Broadcast
    - Matmul 2d
    - Matmul Nd
    - element wise operation(add, sub, div, mul)
    - slicing
    - Dot Product
