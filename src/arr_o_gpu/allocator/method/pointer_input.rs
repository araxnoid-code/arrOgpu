use crate::arr_o_gpu::allocator::Allocator;

impl Allocator {
    pub fn pointer_input(&mut self, data_length: u32) -> (u32, u32) {
        let mut pointer: (u32, u32) = (0, 0);
        if !self.range_space.is_empty() {
            for (idx, space) in self.range_space.iter().enumerate() {
                if let Some(range) = space {
                    let len_space = range.end - range.start;
                    if data_length >= len_space {
                        pointer = (range.start, range.end);
                        self.range_space[idx] = None;
                        break;
                    }
                }

                if (idx as u32) >= data_length - 1 {
                    // using last space
                    pointer = (self.last_space.0, self.last_space.0 + data_length);
                    // update last_space
                    let start = self.last_space.0 + data_length;
                    if start > self.last_space.1 {
                        panic!("Allocator Error: Memory Overflow");
                    }
                    self.last_space.0 = start;
                }
            }
        } else {
            pointer = (self.last_space.0, self.last_space.0 + data_length);
            // update last_space
            let start = self.last_space.0 + data_length;
            if start > self.last_space.1 {
                panic!("Allocator Error: Memory Overflow");
            }
            self.last_space.0 = start;
        }

        pointer
    }
}
