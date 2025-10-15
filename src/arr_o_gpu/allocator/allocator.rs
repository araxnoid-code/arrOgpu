use std::ops::Range;

use uuid::Uuid;

use crate::{ Monagement, MonagementInit };

#[derive(Clone)]
pub struct Allocator {
    pub(crate) range_space: Vec<Option<(u128, usize, Range<u32>)>>,
    pub(crate) empty_idx: Vec<usize>,
    pub(crate) last_space: (u32, u32),
    pub(crate) maximum: u32,

    // monagement branch
    pub monanagement: Monagement,
}

impl Allocator {
    pub fn init(maximum: u32) -> Allocator {
        Self {
            range_space: vec![],
            empty_idx: Vec::new(),
            last_space: (0, maximum),
            maximum,

            // monagement branch
            monanagement: Monagement::init(),
        }
    }

    pub fn range_space(&self) -> &Vec<Option<(u128, usize, Range<u32>)>> {
        &self.range_space
    }

    pub fn last_space(&self) -> (u32, u32) {
        self.last_space
    }

    pub fn empty_idx(&self) -> &Vec<usize> {
        &self.empty_idx
    }
}
