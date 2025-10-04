use crate::ArrOgpuModule;

impl ArrOgpuModule {
    pub fn zero_heap(&mut self) {
        self.allocator.clean_allocate();
    }
}
