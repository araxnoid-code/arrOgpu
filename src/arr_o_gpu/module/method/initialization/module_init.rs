use wgpu::{Adapter, Device, MemoryHints, PowerPreference, Queue};

use crate::{ArrOgpuErr, WgpuLimits};

#[derive(Clone)]
pub enum HeapSize {
    Item(u32),
    Byte(u64),
}

impl HeapSize {
    pub(crate) fn get_size_of_32(&self) -> Result<(u32, u64), ArrOgpuErr> {
        match self {
            HeapSize::Byte(byte) => {
                if byte < &4 {
                    let error =
                        "Error Module Init, length of byte at least equal to or greater than 4";
                    return Err(ArrOgpuErr::ModuleInit(error.to_string()));
                }
                Ok(((*byte as u32) / 4, *byte))
            }
            HeapSize::Item(item) => {
                Ok((*item, (*item as u64) * (std::mem::size_of::<f32>() as u64)))
            }
        }
    }
}

#[derive(Clone)]
pub enum PowerInit {
    HighPerformance,
    LowPower,
}

impl PowerInit {
    pub(crate) fn conversion(&self) -> PowerPreference {
        match &self {
            PowerInit::HighPerformance => PowerPreference::HighPerformance,
            PowerInit::LowPower => PowerPreference::LowPower,
        }
    }
}

#[derive(Clone)]
pub enum Memory {
    Performance,
    MemoryUsage,
}

impl Memory {
    pub(crate) fn conversion(&self) -> MemoryHints {
        match self {
            Self::Performance => MemoryHints::Performance,
            Self::MemoryUsage => MemoryHints::MemoryUsage,
        }
    }
}

#[derive(Clone)]
pub enum WgpuInit {
    ManualDeviceQueue(Adapter, Device, Queue),
    ManualInit(ManualInit),
}

#[derive(Clone)]
pub struct ManualInit {
    pub power: PowerInit,
    pub memory: Memory,
}

impl Default for ManualInit {
    fn default() -> Self {
        Self {
            memory: Memory::MemoryUsage,
            power: PowerInit::LowPower,
        }
    }
}

#[derive(Clone)]
pub struct ArrOgpuModuleInit {
    pub heap_size: HeapSize,
    pub wgpu: WgpuInit,
    pub limits: WgpuLimits,
}

impl Default for ArrOgpuModuleInit {
    fn default() -> Self {
        Self {
            heap_size: HeapSize::Item(100_000),
            wgpu: WgpuInit::ManualInit(ManualInit::default()),
            limits: WgpuLimits::default(),
        }
    }
}
