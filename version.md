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
  - The compute shaders used in execution have all been rewritten and optimized due to the use of new metadata. Compute shaders have now been combined into the compute_shaders folder for easier searching and maintenance, as can be seen in [compute_shaders](./src/arr_o_gpu/compute_shaders/)

- Changes to the method
  - pow has now been split into powi and powf.
  - matmul_nd and matmul_2d have now been merged into matmul, the matmul method itself will automatically determine whether to use matmul nd or matmul 2d compute shaders.
  - The data type bug in the slice method in arrOgpuModule which initially used `SliceRange` has now been changed to `SliceRangeNegativeAble`.

- reduction heap
  - In methods that use parallel reduction (sum, dot_product, sum_axis, sum_axis_keep_dim), additional memory is still used to store the reduction results between workgroups, which in the future will be stored in the main heap if possible.
