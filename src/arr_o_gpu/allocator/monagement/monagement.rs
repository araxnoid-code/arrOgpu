use std::{collections::BTreeMap, ops::Range};

#[derive(Debug, Clone)]
pub struct Monagement_old {
    pub range_space: BTreeMap<u32, (u128, Range<u32>)>,
    //
}
