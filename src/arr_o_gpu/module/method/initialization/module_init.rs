use wgpu::{ Device, MemoryHints, PowerPreference, Queue };

use crate::ArrOgpuErr;

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
            HeapSize::Item(item) =>
                Ok((*item, (*item as u64) * (std::mem::size_of::<f32>() as u64))),
        }
    }
}

pub enum Power {
    HighPerformance,
    LowPower,
}

impl Power {
    pub(crate) fn conversion(&self) -> PowerPreference {
        match &self {
            Power::HighPerformance => PowerPreference::HighPerformance,
            Power::LowPower => PowerPreference::LowPower,
        }
    }
}

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

pub enum WgpuInit {
    ManualDeviceQueue(Device, Queue),
    ManualInit(ManualInit),
}

pub struct ManualInit {
    pub power: Power,
    pub memory: Memory,
}

impl Default for ManualInit {
    fn default() -> Self {
        Self {
            memory: Memory::MemoryUsage,
            power: Power::LowPower,
        }
    }
}

pub struct ArrOgpuModuleInit {
    pub heap_size: HeapSize,
    pub wgpu: WgpuInit,
}

impl Default for ArrOgpuModuleInit {
    fn default() -> Self {
        Self {
            heap_size: HeapSize::Item(100_000),
            wgpu: WgpuInit::ManualInit(ManualInit::default()),
        }
    }
}
