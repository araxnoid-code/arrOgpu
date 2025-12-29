use crate::arr_o_gpu::module::method::array_init::metadata::metadata::ArrayMetadata;

impl ArrayMetadata {
    pub fn pointer(&self) -> [u32; 2] {
        self.pointer
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn shape(&self) -> [u32; 10] {
        self.shape
    }

    pub fn stride(&self) -> [u32; 10] {
        self.stride
    }

    pub fn origin_stride(&self) -> [u32; 10] {
        self.origin_stride
    }
}
