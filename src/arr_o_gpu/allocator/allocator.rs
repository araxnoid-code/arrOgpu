use std::ops::Range;

#[derive(Clone)]
pub struct Allocator {
    pub(crate) range_space: Vec<Option<Range<u32>>>,
    pub(crate) last_space: (u32, u32),
    pub(crate) maximum: u32,
}

impl Allocator {
    pub fn init(maximum: u32) -> Allocator {
        Self {
            range_space: vec![],
            last_space: (0, maximum),
            maximum,
        }
    }

    pub fn range_space(&self) -> &Vec<Option<Range<u32>>> {
        &self.range_space
    }

    pub fn last_space(&self) -> (u32, u32) {
        self.last_space
    }
}
