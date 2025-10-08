#[derive(Debug)]
pub enum SpaceType {
    RangeSpace(u128),
    FragmentSpace(u128, usize),
}
