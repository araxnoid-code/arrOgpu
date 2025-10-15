use uuid::Uuid;

use crate::{ GpuArray, MonagementGet, MonagementInsertRemove, MonagementRemove, SpaceType };

impl Drop for GpuArray {
    fn drop(&mut self) {
        let mut allocator = self.module.allocator.write().unwrap();
        let self_range = self.pointer.0 as u32..self.pointer.1 as u32;

        // monagement branch
        match self.space_type {
            SpaceType::RangeSpace(_) => {
                let _id = Uuid::new_v4().as_u128();

                let end = if let Some(right) = allocator.monanagement.get(&self_range.end) {
                    let end = right.1.end;
                    allocator.monanagement.remove(&self_range.end);
                    end
                } else {
                    self_range.end
                };

                allocator.monanagement.insert(self_range.start, (_id, self_range.start..end));
            }
            SpaceType::FragmentSpace(_id, _idx) => {
                // monagement branch
                let _id = Uuid::new_v4().as_u128();

                let end = if let Some(right) = allocator.monanagement.get(&self_range.end) {
                    let end = right.1.end;
                    allocator.monanagement.remove(&self_range.end);
                    end
                } else {
                    self_range.end
                };

                allocator.monanagement.insert(self_range.start, (_id, self_range.start..end));
            }
        }
        // monagement branch
    }
}
