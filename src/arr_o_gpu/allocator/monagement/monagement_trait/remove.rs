use crate::Monagement_old;

pub trait MonagementRemove {
    fn remove(&mut self, key: &u32) -> Option<(u128, std::ops::Range<u32>)>;
}

impl MonagementRemove for Monagement_old {
    fn remove(&mut self, key: &u32) -> Option<(u128, std::ops::Range<u32>)> {
        self.range_space.remove(key)
    }
}
