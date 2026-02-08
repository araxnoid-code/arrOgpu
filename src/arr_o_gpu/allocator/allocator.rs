use monagement::{Monagement, MonagementInit};

// #[derive(Clone)]
pub struct Allocator {
    pub(crate) maximum: u32,
    pub(crate) core: Monagement,
}

impl Allocator {
    pub fn init(maximum: u32) -> Allocator {
        Self {
            maximum,
            // core update
            core: Monagement::init(MonagementInit::default()).unwrap(),
            // core update
        }
    }

    pub fn core(&self) -> &Monagement {
        &self.core
    }
}
