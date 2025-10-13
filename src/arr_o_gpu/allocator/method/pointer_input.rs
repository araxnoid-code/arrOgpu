use uuid::Uuid;

use crate::{
    arr_o_gpu::allocator::Allocator,
    MonagementIsEmpty,
    MonagementIter,
    MonagementRemove,
    SpaceType,
};

impl Allocator {
    pub fn pointer_input(&mut self, data_length: u32) -> (SpaceType, u32, u32) {
        let mut pointer = (SpaceType::RangeSpace(0), 0, 0);

        // monagement branch

        if !self.monanagement.is_empty() {
            // not empty
            let mut remove_key = None;
            for (key, (_, range)) in self.monanagement.iter() {
                let len = range.end - range.start;
                if data_length == len {
                    // range is same => Range Mode
                    let id = Uuid::new_v4().as_u128();
                    pointer = (SpaceType::RangeSpace(id), range.start, range.end);
                    remove_key = Some(*key);
                    break;
                } else {
                    // range is lower => Fragement Mode

                    break;
                }
            }

            if let Some(key) = remove_key {
                self.monanagement.remove(&key);
            }
        } else {
            // empty

        }

        // monagement branch

        if !self.range_space.is_empty() {
            for (idx, space) in self.range_space.iter().enumerate() {
                if let Some((_, _, range)) = space {
                    let len_space = range.end - range.start;
                    if data_length == len_space {
                        let id = Uuid::new_v4().as_u128();
                        pointer = (SpaceType::RangeSpace(id), range.start, range.end);
                        self.range_space[idx] = None;
                        self.empty_idx.push(idx);
                        break;
                    } else if data_length < len_space {
                        let id = Uuid::new_v4().as_u128();
                        pointer = (
                            SpaceType::FragmentSpace(id, idx),
                            range.start,
                            range.start + data_length,
                        );
                        let update_range = range.start + data_length..range.end;
                        self.range_space[idx] = Some((id, idx, update_range));
                        break;
                    }
                }

                if idx >= self.range_space.len() - 1 {
                    // using last space
                    let id = Uuid::new_v4().as_u128();
                    pointer = (
                        SpaceType::RangeSpace(id),
                        self.last_space.0,
                        self.last_space.0 + data_length,
                    );
                    // update last_space
                    let start = self.last_space.0 + data_length;
                    if start > self.last_space.1 {
                        panic!("Allocator Error: Memory Overflow");
                    }
                    self.last_space.0 = start;
                }
            }
        } else {
            let id = Uuid::new_v4().as_u128();
            pointer = (
                SpaceType::RangeSpace(id),
                self.last_space.0,
                self.last_space.0 + data_length,
            );
            // update last_space
            let start = self.last_space.0 + data_length;
            if start > self.last_space.1 {
                panic!("Allocator Error: Memory Overflow");
            }
            self.last_space.0 = start;
        }

        pointer
    }
}
