use std::{ops::Range, u128};

use crate::Monagement;

pub trait MonagementInsertRemove {
    fn insert(&mut self, key: u32, range: (u128, Range<u32>)) -> Option<(u128, Range<u32>)>;
}

impl MonagementInsertRemove for Monagement {
    fn insert(&mut self, key: u32, range: (u128, Range<u32>)) -> Option<(u128, Range<u32>)> {
        self.range_space.insert(key, range)
    }
}
