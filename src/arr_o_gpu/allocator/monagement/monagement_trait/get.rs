use crate::Monagement;

pub trait MonagementGet {
    fn get(&self, key: &u32) -> Option<&(u128, std::ops::Range<u32>)>;

    fn get_mut(&mut self, key: &u32) -> Option<&mut (u128, std::ops::Range<u32>)>;
}

impl MonagementGet for Monagement {
    fn get(&self, key: &u32) -> Option<&(u128, std::ops::Range<u32>)> {
        self.range_space.get(key)
    }

    fn get_mut(&mut self, key: &u32) -> Option<&mut (u128, std::ops::Range<u32>)> {
        self.range_space.get_mut(key)
    }
}
