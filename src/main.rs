use std::{ cell::RefCell, collections::VecDeque, ops::Range, rc::Rc, sync::{ Mutex, RwLock } };

fn main() {
    // let module = ArrOgpuModule::default();

    // println!("{:?} (1)", module.allocator_read().range_space());
    // {
    //     let _a = module.array_from_vector(
    //         &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    //         &[1, 10]
    //     );
    //     // let _b = module.array_from_vector(&[11.0, 12.0, 13.0, 14.0], &[1, 4]);
    //     println!("{:?} (2)", module.allocator_read().range_space());
    // }
    // println!("{:?} (3)", module.allocator_read().range_space());

    // {
    //     let _a = module.array_from_vector(&[51.0, 52.0, 53.0, 54.0], &[1, 4]);
    //     println!("{:?} (4)", module.allocator_read().range_space());
    //     let _b = module.array_from_vector(&[101.0, 102.0, 103.0], &[1, 3]);
    //     println!("{:?} (4)", module.allocator_read().range_space());
    //     // let _b = module.array_from_vector(&[100.0, 101.0], &[1, 2]);
    //     // println!("{:?} (5)", module.allocator_read().range_space());
    // }
    // println!("{:?} (6)", module.allocator_read().range_space());

    // println!("\n{:?}", module.get_heap());

    let mut ranges: VecDeque<Range<usize>> = VecDeque::new();
    let mut indexs: VecDeque<usize> = VecDeque::new();

    allocator_prototype(&mut ranges, &mut indexs, 0..1, None, None);
    // allocator_prototype(&mut ranges, &mut indexs, 2..6, None, None);
    // allocator_prototype(&mut ranges, &mut indexs, 10..13, None, None);
    // allocator_prototype(&mut ranges, &mut indexs, 6..10, None, None);

    println!("{:?}", indexs);
}

pub fn allocator_prototype(
    ranges: &mut VecDeque<Range<usize>>,
    indexs: &mut VecDeque<usize>,
    input: Range<usize>,
    right: Option<usize>,
    left: Option<usize>
) {
    if ranges.is_empty() {
        indexs.push_back(input.start);
        ranges.push_back(input);
    } else {
        let mut right = right.unwrap_or(0);
        let mut left = left.unwrap_or(indexs.len() - 1);
        let target = input.start;

        if right == left {
            let pivot = indexs[right];
            if target > pivot {
                indexs.insert(right + 1, target);
            } else {
                indexs.insert(right, target);
            }
        } else {
            let idx = right + ((((left - right) as f64) / 2.0) as usize);
            let pivot = indexs[idx];

            if target > pivot {
                let n_idx = idx + 1;
                right = if n_idx > indexs.len() { idx } else { n_idx };
                allocator_prototype(ranges, indexs, input, Some(right), Some(left));
            } else {
                let n_idx = (idx as i32) + 1;
                left = if n_idx < 0 { idx } else { n_idx as usize };
                allocator_prototype(ranges, indexs, input, Some(right), Some(left));
            }
        }
    }
}
