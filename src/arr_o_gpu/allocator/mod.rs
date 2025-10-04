mod method;
use std::ops::Range;

pub struct Allocator {
    pub range_space: Vec<Option<Range<u32>>>,
    pub last_space: (u32, u32),
    pub maximum: u32,
}

impl Allocator {
    pub fn init(maximum: u32) -> Allocator {
        Self {
            range_space: vec![],
            last_space: (0, maximum),
            maximum,
        }
    }
}
