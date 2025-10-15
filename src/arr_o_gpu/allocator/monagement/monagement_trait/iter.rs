use crate::Monagement;

pub(crate) trait MonagementIter {
    fn iter(&self) -> std::collections::btree_map::Iter<'_, u32, (u128, std::ops::Range<u32>)>;

    fn iter_mut(
        &mut self
    ) -> std::collections::btree_map::IterMut<'_, u32, (u128, std::ops::Range<u32>)>;
}

impl MonagementIter for Monagement {
    fn iter(&self) -> std::collections::btree_map::Iter<'_, u32, (u128, std::ops::Range<u32>)> {
        self.range_space.iter()
    }

    fn iter_mut(
        &mut self
    ) -> std::collections::btree_map::IterMut<'_, u32, (u128, std::ops::Range<u32>)> {
        self.range_space.iter_mut()
    }
}
