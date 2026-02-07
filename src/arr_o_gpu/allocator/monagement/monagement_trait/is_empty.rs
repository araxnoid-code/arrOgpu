use crate::Monagement;

pub trait MonagementIsEmpty {
    fn is_empty(&self) -> bool;
}

impl MonagementIsEmpty for Monagement {
    fn is_empty(&self) -> bool {
        self.range_space.is_empty()
    }
}
