use std::collections::BTreeMap;

use crate::Monagement_old;

pub trait MonagementInit_old {
    fn init() -> Monagement_old;
}

impl MonagementInit_old for Monagement_old {
    fn init() -> Monagement_old {
        Self {
            range_space: BTreeMap::new(),
        }
    }
}
