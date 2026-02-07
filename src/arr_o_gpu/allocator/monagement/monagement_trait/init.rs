use std::collections::BTreeMap;

use crate::Monagement;

pub trait MonagementInit {
    fn init() -> Monagement;
}

impl MonagementInit for Monagement {
    fn init() -> Monagement {
        Self {
            range_space: BTreeMap::new(),
        }
    }
}
