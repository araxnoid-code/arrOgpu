// FUNCTION
pub(crate) const ABS_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/abs/abs_contiguous.wgsl");
pub(crate) const ABS_VIEW_SHADERS_PATH: &'static str = include_str!("./function/abs/abs_view.wgsl");

pub(crate) const LOG2_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/log2/log2_contiguous.wgsl");
pub(crate) const LOG2_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/log2/log2_view.wgsl");

pub(crate) const POWF_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/powf/powf_contiguous.wgsl");
pub(crate) const POWF_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/powf/powf_view.wgsl");

pub(crate) const POWI_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/powi/powi_contiguous.wgsl");
pub(crate) const POWI_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/powi/powi_view.wgsl");

pub(crate) const SQRT_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/sqrt/sqrt_contiguous.wgsl");
pub(crate) const SQRT_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/sqrt/sqrt_view.wgsl");

pub(crate) const SIN_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/sin_cos_tan/sin/sin_contiguous.wgsl");
pub(crate) const SIN_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/sin_cos_tan/sin/sin_view.wgsl");

pub(crate) const COS_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/sin_cos_tan/cos/cos_contiguous.wgsl");
pub(crate) const COS_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/sin_cos_tan/cos/cos_view.wgsl");

pub(crate) const TAN_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./function/sin_cos_tan/tan/tan_contiguous.wgsl");
pub(crate) const TAN_VIEW_SHADERS_PATH: &'static str =
    include_str!("./function/sin_cos_tan/tan/tan_view.wgsl");

// Operation
pub(crate) const MATMUL_2D_SHADERS_PATH: &'static str =
    include_str!("./operation/matmul/matmul_2d.wgsl");
pub(crate) const MATMUL_ND_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./operation/matmul/matmul_nd_contiguous.wgsl");
pub(crate) const MATMUL_ND_VIEW_SHADERS_PATH: &'static str =
    include_str!("./operation/matmul/matmul_nd_view.wgsl");

pub(crate) const DOT_PRODUCT_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./operation/dot_product/dot_product_contiguous.wgsl");
pub(crate) const DOT_PRODUCT_VIEW_SHADERS_PATH: &'static str =
    include_str!("./operation/dot_product/dot_product_view.wgsl");

pub(crate) const SUM_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./operation/sum/sum_contiguous.wgsl");
pub(crate) const SUM_VIEW_SHADERS_PATH: &'static str =
    include_str!("./operation/sum/sum_view.wgsl");

pub(crate) const SUM_AXIS_SHADERS_PATH: &'static str =
    include_str!("./operation/sum_axis/sum_axis.wgsl");
pub(crate) const SUM_AXIS_KEEP_DIM_SHADERS_PATH: &'static str =
    include_str!("./operation/sum_axis/sum_axis_keep_dim.wgsl");

pub(crate) const ADD_NON_SCALAR_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./operation/element_wise/add/non_scalar/add_contiguous.wgsl");
pub(crate) const ADD_NON_SCALAR_VIEW_SHADERS_PATH: &'static str =
    include_str!("./operation/element_wise/add/non_scalar/add_view.wgsl");
pub(crate) const ADD_SCALAR_CONTIGUOUS_SHADERS_PATH: &'static str =
    include_str!("./operation/element_wise/add/scalar/add_scalar_contiguous.wgsl");
pub(crate) const ADD_SCALAR_VIEW_SHADERS_PATH: &'static str =
    include_str!("./operation/element_wise/add/scalar/add_scalar_view.wgsl");

// VIEW
pub(crate) const CONTIGUOUS_SHADERS_PATH: &'static str = include_str!("./view/contiguous.wgsl");
