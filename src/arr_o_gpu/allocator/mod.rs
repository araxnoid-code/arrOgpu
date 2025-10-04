mod method;
pub use method::*;
use std::ops::Range;

pub struct Allocator {
    range_space: Vec<Option<Range<u32>>>,
    last_space: (u32, u32),
    maximum: u32,
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
