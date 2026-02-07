use crate::Monagement;

pub trait MonagementRemove {
    fn remove(&mut self, key: &u32) -> Option<(u128, std::ops::Range<u32>)>;
}

impl MonagementRemove for Monagement {
    fn remove(&mut self, key: &u32) -> Option<(u128, std::ops::Range<u32>)> {
        self.range_space.remove(key)
    }
}
