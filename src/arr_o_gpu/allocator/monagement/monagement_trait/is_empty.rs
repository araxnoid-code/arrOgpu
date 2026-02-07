use crate::Monagement_old;

pub trait MonagementIsEmpty {
    fn is_empty(&self) -> bool;
}

impl MonagementIsEmpty for Monagement_old {
    fn is_empty(&self) -> bool {
        self.range_space.is_empty()
    }
}
