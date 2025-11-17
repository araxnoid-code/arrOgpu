use std::{
    fmt::Debug,
    ops::{ Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive },
};

#[derive(Clone)]
pub struct SliceRange {
    pub start: Option<usize>,
    pub end: Option<usize>,
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
    fn start(&self) -> Option<usize>;
    fn end(&self) -> Option<usize>;
}

impl SlicingRangeTrait for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl SlicingRangeTrait for RangeFull {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl SlicingRangeTrait for RangeFrom<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl SlicingRangeTrait for RangeTo<usize> {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl SlicingRangeTrait for RangeInclusive<usize> {
    fn start(&self) -> Option<usize> {
        Some(*self.start())
    }

    fn end(&self) -> Option<usize> {
        Some(*self.end())
    }
}

impl SlicingRangeTrait for RangeToInclusive<usize> {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}
