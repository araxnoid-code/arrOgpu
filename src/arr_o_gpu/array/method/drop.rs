use std::ops::Range;

use uuid::Uuid;

use crate::{ GpuArray, SpaceType };

impl Drop for GpuArray {
    fn drop(&mut self) {
        let mut allocator = self.module.allocator.write().unwrap();
        let self_range = self.pointer.0 as u32..self.pointer.1 as u32;

        match self.space_type {
            SpaceType::RangeSpace(_) => {
                range_space_handle(allocator, self_range);
            }
            SpaceType::FragmentSpace(id, idx) => {
                if let Some(range_space) = allocator.range_space.get(idx) {
                    if let Some((range_id, _, range)) = range_space {
                        if &id == range_id {
                            let array_range = self.pointer;
                            // new range
                            let start = array_range.0 as u32;
                            let end = range.end;
                            let new_range = start..end;
                            allocator.range_space[idx].as_mut().unwrap().2 = new_range;
                        } else {
                            range_space_handle(allocator, self_range);
                        }
                    } else {
                        range_space_handle(allocator, self_range);
                    }
                } else {
                    range_space_handle(allocator, self_range);
                }
            }
        }
    }
}

fn range_space_handle(
    mut allocator: std::sync::RwLockWriteGuard<'_, crate::Allocator>,
    range: Range<u32>
) {
    if allocator.empty_idx.is_empty() {
        let index = allocator.range_space.len();
        let id = Uuid::new_v4().as_u128();
        let range_space_element = (id, index, range);
        allocator.range_space.push(Some(range_space_element));
    } else {
        let id = Uuid::new_v4().as_u128();
        let idx = allocator.empty_idx.pop().unwrap();
        let range_space_element = (id, idx, range);
        allocator.range_space[idx] = Some(range_space_element);
    }
}
