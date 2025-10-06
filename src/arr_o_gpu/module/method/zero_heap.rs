use crate::ArrOgpuModule;

impl ArrOgpuModule {
    pub fn zero_heap(&mut self) {
        self.allocator.write().unwrap().clean_allocate();
    }
}
