use std::{
    fmt::Debug,
    ops::{ Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive },
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

pub fn r<T: SlicingRangeTrait>(range: T) -> SliceRange {
    SliceRange {
        start: range.start(),
        end: range.end(),
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
