use monagement::{Monagement, MonagementInit};
use std::ops::Range;

// #[derive(Clone)]
pub struct Allocator {
    pub(crate) maximum: u32,
    pub(crate) core: Monagement,
}

impl Allocator {
    pub fn init(maximum: u32) -> Allocator {
        Self {
            // range_space: vec![],
            // empty_idx: Vec::new(),
            // last_space: (0, maximum),
            maximum,
            // core update
            core: Monagement::init(MonagementInit::default()).unwrap(),
            // core update
        }
    }
}
