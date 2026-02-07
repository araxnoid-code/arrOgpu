use std::{collections::BTreeMap, ops::Range};

#[derive(Debug, Clone)]
pub struct Monagement {
    pub range_space: BTreeMap<u32, (u128, Range<u32>)>,
}
