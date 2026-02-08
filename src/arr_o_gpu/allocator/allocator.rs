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
            core: Monagement::init(MonagementInit {
                maximum: maximum as u64,
                ..Default::default()
            })
            .unwrap(),
        }
    }

    pub fn core(&self) -> &Monagement {
        &self.core
    }
}
