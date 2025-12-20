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