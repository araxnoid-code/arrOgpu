use std::num::NonZeroU64;

use uuid::Uuid;

use crate::arr_o_gpu::allocator::Allocator;

impl Allocator {
    pub fn allocate(&mut self, size: u32) -> Result<monagement::Allocated, String> {
        self.core
            .allocate(NonZeroU64::new(size as u64).ok_or("size must be greater than 0")?)
    }

    // pub fn allocate(&mut self, data_length: u32) -> (u32, u32) {
    //     let mut pointer = (SpaceType::RangeSpace(0), 0, 0);

    //     // monagement branch
    //     if !self.monanagement.is_empty() {
    //         // not empty
    //         for (_, (key, (_, range))) in self.monanagement.iter().enumerate() {
    //             let len = range.end - range.start;
    //             if data_length == len {
    //                 // range is same => Range Mode
    //                 let id = Uuid::new_v4().as_u128();
    //                 pointer = (SpaceType::RangeSpace(id), range.start, range.end);

    //                 let key = *key;
    //                 self.monanagement.remove(&key);
    //                 break;
    //             } else if data_length < len {
    //                 // range is bigger => Fragement Mode
    //                 let id = Uuid::new_v4().as_u128();
    //                 pointer = (
    //                     SpaceType::FragmentSpace(id, 0),
    //                     range.start,
    //                     range.start + data_length,
    //                 );

    //                 // update range
    //                 let key = *key;
    //                 let start = range.start + data_length;
    //                 let end = range.end;

    //                 self.monanagement.remove(&key);
    //                 self.monanagement.insert(start, (id, start..end));
    //                 break;
    //             }
    //         }

    //         if pointer.1 == 0 && pointer.2 == 0 {
    //             // using last space
    //             let id = Uuid::new_v4().as_u128();
    //             pointer = (
    //                 SpaceType::RangeSpace(id),
    //                 self.last_space.0,
    //                 self.last_space.0 + data_length,
    //             );
    //             // update last_space
    //             let start = self.last_space.0 + data_length;
    //             if start > self.last_space.1 {
    //                 panic!("Allocator Error: Memory Overflow");
    //             }
    //             self.last_space.0 = start;
    //         }
    //     } else {
    //         // empty
    //         let id = Uuid::new_v4().as_u128();

    //         pointer = (
    //             SpaceType::RangeSpace(id),
    //             self.last_space.0,
    //             self.last_space.0 + data_length,
    //         );
    //         // update last_space
    //         let start = self.last_space.0 + data_length;
    //         if start > self.last_space.1 {
    //             panic!("Allocator Error: Memory Overflow");
    //         }
    //         self.last_space.0 = start;
    //     }
    //     // monagement branch

    //     pointer
    // }
}
