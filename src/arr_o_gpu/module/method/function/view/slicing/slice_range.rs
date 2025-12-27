use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

#[derive(Clone)]
pub struct SliceRange {
    pub start: Option<u32>,
    pub end: Option<u32>,
}

impl Debug for SliceRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start = match self.start {
            Some(start) => format!("{}", start),
            None => "".to_string(),
        };

        let end = match self.end {
            Some(end) => format!("{}", end),
            None => "".to_string(),
        };
        let text = format!("({}..{})", start, end);
        f.write_str(&text)
    }
}

pub trait SlicingRangeTrait {
    fn start(&self) -> Option<u32>;
    fn end(&self) -> Option<u32>;
}

impl SlicingRangeTrait for Range<u32> {
    fn start(&self) -> Option<u32> {
        Some(self.start)
    }

    fn end(&self) -> Option<u32> {
        Some(self.end)
    }
}

impl SlicingRangeTrait for RangeFull {
    fn start(&self) -> Option<u32> {
        None
    }

    fn end(&self) -> Option<u32> {
        None
    }
}

impl SlicingRangeTrait for RangeFrom<u32> {
    fn start(&self) -> Option<u32> {
        Some(self.start)
    }

    fn end(&self) -> Option<u32> {
        None
    }
}

impl SlicingRangeTrait for RangeTo<u32> {
    fn start(&self) -> Option<u32> {
        None
    }

    fn end(&self) -> Option<u32> {
        Some(self.end)
    }
}

impl SlicingRangeTrait for RangeInclusive<u32> {
    fn start(&self) -> Option<u32> {
        Some(*self.start())
    }

    fn end(&self) -> Option<u32> {
        Some(*self.end())
    }
}

impl SlicingRangeTrait for RangeToInclusive<u32> {
    fn start(&self) -> Option<u32> {
        None
    }

    fn end(&self) -> Option<u32> {
        Some(self.end)
    }
}

//
#[derive(Debug)]
pub struct SliceRangeNegativeAble {
    pub start: Option<i32>,
    pub end: Option<i32>,
}

pub fn r<T: SlicingRangeTraitNegativeAble>(range: T) -> SliceRangeNegativeAble {
    SliceRangeNegativeAble {
        start: range.start(),
        end: range.end(),
    }
}

pub trait SlicingRangeTraitNegativeAble {
    fn start(&self) -> Option<i32>;
    fn end(&self) -> Option<i32>;
}

impl SlicingRangeTraitNegativeAble for Range<i32> {
    fn start(&self) -> Option<i32> {
        Some(self.start)
    }

    fn end(&self) -> Option<i32> {
        Some(self.end)
    }
}

impl SlicingRangeTraitNegativeAble for RangeFull {
    fn start(&self) -> Option<i32> {
        None
    }

    fn end(&self) -> Option<i32> {
        None
    }
}

impl SlicingRangeTraitNegativeAble for RangeFrom<i32> {
    fn start(&self) -> Option<i32> {
        Some(self.start)
    }

    fn end(&self) -> Option<i32> {
        None
    }
}

impl SlicingRangeTraitNegativeAble for RangeTo<i32> {
    fn start(&self) -> Option<i32> {
        None
    }

    fn end(&self) -> Option<i32> {
        Some(self.end)
    }
}

impl SlicingRangeTraitNegativeAble for RangeInclusive<i32> {
    fn start(&self) -> Option<i32> {
        Some(*self.start())
    }

    fn end(&self) -> Option<i32> {
        Some(*self.end())
    }
}

impl SlicingRangeTraitNegativeAble for RangeToInclusive<i32> {
    fn start(&self) -> Option<i32> {
        None
    }

    fn end(&self) -> Option<i32> {
        Some(self.end)
    }
}
